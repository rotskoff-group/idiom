#!/usr/bin/env python3
"""
Extract all HDF5 records whose UniProt accession (dropping any “-…” suffix)
matches an accession in a large FASTA, writing all fields into one output HDF5.
This version bulk-loads each dataset once per shard, then filters in memory.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py

# ──────────────────────────────────────────────────────────────────────────────
BUF_SIZE = 500_000                   # flush buffer every this many rows
# ──────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()

def parse_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    current = None
    with path.open("rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                seqs[current] = []
            else:
                if current is None:
                    raise ValueError(f"Malformed FASTA before line: {line}")
                seqs[current].append(line)
    return {acc: "".join(parts) for acc, parts in seqs.items()}

def extend_datasets(dsets: Dict[str, h5py.Dataset],
                    offset: int,
                    buffers: Dict[str, List]) -> int:
    n_new = len(next(iter(buffers.values())))
    if n_new == 0:
        return offset
    for name, ds in dsets.items():
        ds.resize(offset + n_new, axis=0)
        ds[offset: offset + n_new] = buffers[name]
    return offset + n_new

def main():
    parser = argparse.ArgumentParser(
        description="Bulk-load HDF5 shards and extract records by accession ID."
    )
    parser.add_argument("--fasta",  required=True, type=Path,
                        help="FASTA file (≫10 M sequences).")
    parser.add_argument("--h5_dir", required=True, type=Path,
                        help="Directory with part_*.h5 shards.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output HDF5 file for matched records.")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"

    # 1) Read FASTA → set of accessions
    log("Parsing FASTA …")
    t0 = time.perf_counter()
    fasta_map   = parse_fasta(args.fasta)
    target_accs = set(fasta_map.keys())
    log(f"  • {len(target_accs):,} unique accessions loaded in {time.perf_counter() - t0:.1f}s")

    # 2) Find shards
    shards = sorted(p for p in args.h5_dir.glob("part_*.h5") if p.is_file())
    if not shards:
        log(f"ERROR: no shards in {args.h5_dir}")
        sys.exit(1)
    log(f"Discovered {len(shards)} shard files")

    # 3) Inspect first shard for schema
    with h5py.File(shards[0], "r") as tpl:
        ds_names = list(tpl.keys())
        shapes   = {n: tpl[n].shape[1:] for n in ds_names}
        dtypes   = {n: tpl[n].dtype     for n in ds_names}
        chunks   = {n: tpl[n].chunks    for n in ds_names}

    # 4) Create output HDF5
    log(f"Creating output file {args.output} …")
    with h5py.File(args.output, "w", libver="latest") as fout:
        out_ds = {}
        for n in ds_names:
            maxshape = (None,) + shapes[n]
            chunking = chunks[n] or ((BUF_SIZE,) + shapes[n])
            out_ds[n] = fout.create_dataset(
                name     = n,
                shape    = (0,) + shapes[n],
                maxshape = maxshape,
                dtype    = dtypes[n],
                chunks   = chunking
            )

        buffers = {n: [] for n in ds_names}
        written = 0
        matched = 0

        # 5) Process each shard
        for idx, shard_path in enumerate(shards, start=1):
            t_shard = time.perf_counter()
            log(f"[{idx}/{len(shards)}] Loading {shard_path.name} …")

            with h5py.File(shard_path, "r") as h5:
                total_rows = h5[ds_names[0]].shape[0]

                # Bulk-read all columns once
                data_arrays: Dict[str, List] = {}
                for name in ds_names:
                    arr = h5[name][:]
                    raw = arr.tolist()
                    # Normalize any bytes → str
                    data_arrays[name] = [
                        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
                        for x in raw
                    ]

                kept_in_shard = 0

                # Filter in-memory
                for i, raw_id in enumerate(data_arrays["accession_ids"]):
                    base = raw_id.split("-", 1)[0]
                    if base not in target_accs:
                        continue

                    # buffer every column’s i-th element
                    for name in ds_names:
                        buffers[name].append(data_arrays[name][i])

                    kept_in_shard += 1
                    matched         += 1

                    # flush when buffer is big
                    if kept_in_shard and kept_in_shard % BUF_SIZE == 0:
                        written = extend_datasets(out_ds, written, buffers)
                        log(f"  • Flushed {BUF_SIZE:,} rows (total written: {written:,})")
                        buffers = {n: [] for n in ds_names}

            elapsed = time.perf_counter() - t_shard
            log(f"Completed {shard_path.name}: kept {kept_in_shard:,}/{total_rows:,} rows in {elapsed:.1f}s")

        # 6) Final flush
        rem = len(next(iter(buffers.values())))
        if rem:
            written = extend_datasets(out_ds, written, buffers)
            log(f"Final flush of {rem:,} rows (grand total {written:,})")

    # 7) Done
    log(f"ALL DONE – wrote {matched:,} matched records to {args.output}")

if __name__ == "__main__":
    main()
