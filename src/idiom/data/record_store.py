"""Memory-mapped columnar record store built from a record FASTA.

A store is a directory of flat binary files — contiguous sequence and accession byte buffers, CSR
offset arrays, and IDR coordinate arrays — read back with numpy.memmap and indexed to a Record on
demand. The record set is identical to idiom.data.io.read_records, which the builder consumes.

By default a store is a sidecar named "<fasta>.idiomstore", built on first use by open_or_build or
ahead of time with the idiom_build_store CLI:

    idiom_build_store --fasta /path/train.fasta            # -> /path/train.fasta.idiomstore
"""

from __future__ import annotations

import array
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from loguru import logger as log

from idiom.data.io import Record, read_records

STORE_SUFFIX = ".idiomstore"
_VERSION = 1
_META = "meta.json"
_OFF_DTYPE = np.int64       # CSR offsets into the byte buffers (file can exceed 2 GB)
_COORD_DTYPE = np.int32     # IDR coords (<= max protein length, comfortably in int32)
_BYTE_DTYPE = np.uint8


def store_path_for(fasta: str | Path) -> Path:
    """Return the default sidecar store directory for a record FASTA.

    Args:
        fasta (str | Path): Path to the record FASTA.

    Returns:
        Path: The path "<fasta>.idiomstore".
    """
    return Path(str(fasta) + STORE_SUFFIX)


def _source_sig(fasta: str | Path) -> dict:
    st = Path(fasta).stat()
    return {"source": str(Path(fasta).resolve()), "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns}


def build_record_store(
    fasta: str | Path, store_dir: str | Path | None = None, *, drop_noncanonical: bool = True
) -> Path:
    """Parse a record FASTA into a columnar store and return its directory.

    The store is built into a temporary directory and atomically renamed into place. Peak memory is
    dominated by read_records, which materializes the whole FASTA; the index arrays add about 24
    bytes per record and the byte buffers are streamed to disk.

    Args:
        fasta (str | Path): Path to the record FASTA to convert.
        store_dir (str | Path | None): Output store directory; "<fasta>.idiomstore" if None.
        drop_noncanonical (bool): If True, drop non-canonical sequences, as read_records does.

    Returns:
        Path: The store directory.
    """
    fasta = Path(fasta)
    store_dir = Path(store_dir) if store_dir else store_path_for(fasta)
    tmp = store_dir.with_name(store_dir.name + f".tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    seq_off = array.array("q", [0])  # int64; seq i = seq_data[seq_off[i]:seq_off[i+1]]
    acc_off = array.array("q", [0])
    idr_start = array.array("i")
    idr_end = array.array("i")
    n, so, ao = 0, 0, 0
    with open(tmp / "seq_data.bin", "wb") as sf, open(tmp / "acc_data.bin", "wb") as af:
        for rec in read_records(fasta, drop_noncanonical=drop_noncanonical):
            sb = rec.full_seq.encode("ascii")
            ab = rec.accession.encode("ascii")
            sf.write(sb)
            af.write(ab)
            so += len(sb)
            ao += len(ab)
            seq_off.append(so)
            acc_off.append(ao)
            idr_start.append(rec.idr_start)
            idr_end.append(rec.idr_end)
            n += 1

    np.asarray(seq_off, dtype=_OFF_DTYPE).tofile(tmp / "seq_off.bin")
    np.asarray(acc_off, dtype=_OFF_DTYPE).tofile(tmp / "acc_off.bin")
    np.asarray(idr_start, dtype=_COORD_DTYPE).tofile(tmp / "idr_start.bin")
    np.asarray(idr_end, dtype=_COORD_DTYPE).tofile(tmp / "idr_end.bin")
    meta = {
        "version": _VERSION, "n": n, "seq_bytes": so, "acc_bytes": ao,
        "drop_noncanonical": drop_noncanonical, **_source_sig(fasta),
    }
    (tmp / _META).write_text(json.dumps(meta, indent=2))

    if store_dir.exists():
        shutil.rmtree(store_dir)
    os.replace(tmp, store_dir)  # atomic on the same filesystem
    log.info(f"built record store: {n:,} records -> {store_dir}")
    return store_dir


class RecordStore:
    """Read-only, memory-mapped view over a built store.

    Attributes:
        dir (Path): The store directory.
        meta (dict): The store's meta.json contents.
    """

    def __init__(self, store_dir: str | Path) -> None:
        """Open a built store and memory-map its arrays.

        Args:
            store_dir (str | Path): Directory holding the store files.
        """
        self.dir = Path(store_dir)
        self.meta = json.loads((self.dir / _META).read_text())
        n = self.meta["n"]
        self._seq = self._mmap("seq_data.bin", _BYTE_DTYPE, self.meta["seq_bytes"])
        self._acc = self._mmap("acc_data.bin", _BYTE_DTYPE, self.meta["acc_bytes"])
        self._seq_off = self._mmap("seq_off.bin", _OFF_DTYPE, n + 1)
        self._acc_off = self._mmap("acc_off.bin", _OFF_DTYPE, n + 1)
        self._idr_start = self._mmap("idr_start.bin", _COORD_DTYPE, n)
        self._idr_end = self._mmap("idr_end.bin", _COORD_DTYPE, n)

    def _mmap(self, name: str, dtype, count: int) -> np.ndarray:
        if count == 0:  # np.memmap rejects empty files
            return np.empty(0, dtype=dtype)
        return np.memmap(self.dir / name, dtype=dtype, mode="r", shape=(count,))

    def __len__(self) -> int:
        return self.meta["n"]

    def seq_lengths(self) -> np.ndarray:
        """Return the full_seq length of every record.

        Returns:
            np.ndarray: An int64 array of per-record sequence lengths.
        """
        return (self._seq_off[1:] - self._seq_off[:-1]).astype(np.int64)

    def __getitem__(self, i: int) -> Record:
        """Decode one record from the memory-mapped buffers.

        Args:
            i (int): Record index.

        Returns:
            Record: The record at index i.
        """
        s, e = int(self._seq_off[i]), int(self._seq_off[i + 1])
        a0, a1 = int(self._acc_off[i]), int(self._acc_off[i + 1])
        seq = self._seq[s:e].tobytes().decode("ascii")
        acc = self._acc[a0:a1].tobytes().decode("ascii")
        return Record(acc, seq, int(self._idr_start[i]), int(self._idr_end[i]))


def _valid(store_dir: Path, fasta: Path) -> bool:
    try:
        meta = json.loads((store_dir / _META).read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return False
    if meta.get("version") != _VERSION:
        return False
    sig = _source_sig(fasta)  # rebuild if the source FASTA changed
    return meta.get("source_size") == sig["source_size"] and meta.get("source_mtime_ns") == sig["source_mtime_ns"]


def open_or_build(
    fasta: str | Path, *, drop_noncanonical: bool = True, lock_timeout: float = 3600.0, poll: float = 2.0
) -> RecordStore:
    """Return a RecordStore for fasta, building the sidecar store if it is missing or stale.

    A store is stale when its recorded version, source size, or source mtime no longer match the
    FASTA. Concurrent callers coordinate through an O_EXCL lock file: one builds while the others
    poll and then open the result. A lock older than lock_timeout is treated as abandoned and
    removed.

    Args:
        fasta (str | Path): Path to the record FASTA.
        drop_noncanonical (bool): If True, drop non-canonical sequences when building.
        lock_timeout (float): Seconds after which a lock is considered stale, and the limit on how
            long a waiting caller blocks.
        poll (float): Seconds between checks while waiting for another process to finish building.

    Returns:
        RecordStore: A memory-mapped store for fasta.

    Raises:
        TimeoutError: If the build lock is not released within lock_timeout.
    """
    fasta = Path(fasta)
    store_dir = store_path_for(fasta)
    lock = store_dir.with_name(store_dir.name + ".lock")
    deadline = time.time() + lock_timeout
    while True:
        if _valid(store_dir, fasta):
            return RecordStore(store_dir)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:  # break a stale lock from a crashed builder
                if time.time() - lock.stat().st_mtime > lock_timeout:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise TimeoutError(f"timed out waiting for record store build: {store_dir}")
            time.sleep(poll)
            continue
        try:
            if not _valid(store_dir, fasta):
                build_record_store(fasta, store_dir, drop_noncanonical=drop_noncanonical)
            return RecordStore(store_dir)
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)


def main() -> None:
    """Run the idiom_build_store CLI, building a record store from a FASTA and printing its size."""
    import argparse

    ap = argparse.ArgumentParser(description="Build a memory-mapped record store from a record FASTA.")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", help="store directory (default: <fasta>.idiomstore)")
    ap.add_argument("--keep-noncanonical", action="store_true", help="do not drop non-canonical sequences")
    args = ap.parse_args()
    out = build_record_store(args.fasta, args.out, drop_noncanonical=not args.keep_noncanonical)
    store = RecordStore(out)
    print(f"built {len(store):,} records -> {out}")


if __name__ == "__main__":
    main()
