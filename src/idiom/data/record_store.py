"""Memory-mapped columnar record store — fast, low-RAM, shared-across-ranks dataset backing.

``read_records`` parses a multi-GB record FASTA into tens of millions of frozen :class:`Record`
objects; under DDP **every rank repeats that parse**, so startup is minutes and RAM is
``n_ranks × tens of GB`` (the v2 53.6M-record train split is ~58 GB *per process*). This module
converts a record FASTA **once** into a columnar on-disk store — contiguous sequence/accession byte
buffers + CSR offset arrays + IDR-coord arrays — read back via :func:`numpy.memmap`: lazy,
near-instant to open, and shared across processes through the OS page cache (one copy regardless of
rank count).

Behavior is **identical to** :func:`idiom.data.io.read_records`: the builder consumes
``read_records``, so the non-canonical / malformed-header / out-of-range drops are exactly the same.

Build once, then training/SFT just point at the same FASTA (the store is a sidecar, auto-built and
DDP-safe). Pre-build big splits with the ``idiom_build_store`` CLI so launches start in seconds:

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
    """Default sidecar store directory for a record FASTA (``<fasta>.idiomstore``)."""
    return Path(str(fasta) + STORE_SUFFIX)


def _source_sig(fasta: str | Path) -> dict:
    st = Path(fasta).stat()
    return {"source": str(Path(fasta).resolve()), "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns}


def build_record_store(
    fasta: str | Path, store_dir: str | Path | None = None, *, drop_noncanonical: bool = True
) -> Path:
    """Parse ``fasta`` once (via :func:`read_records`) into a columnar store; return its directory.

    Writes the byte buffers by streaming; only the small offset/coord index arrays are held in RAM
    (~24 B/record). Builds into a temp dir and atomically renames, so a store dir is always complete.
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
    """Read-only, memory-mapped view over a built store. Indexes to :class:`Record` lazily."""

    def __init__(self, store_dir: str | Path) -> None:
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
        """Per-record ``full_seq`` length, vectorized (for fast length filtering)."""
        return (self._seq_off[1:] - self._seq_off[:-1]).astype(np.int64)

    def __getitem__(self, i: int) -> Record:
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
    """Return a :class:`RecordStore` for ``fasta``, building the sidecar store if missing/stale.

    DDP-safe: an ``O_EXCL`` lock file ensures exactly one rank builds while the others wait and then
    mmap the result. A lock older than ``lock_timeout`` (a crashed builder) is broken.
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
