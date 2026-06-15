#!/usr/bin/env python3
"""
Create fill-in-the-middle (FIM) representations of IDRs for 40 M proteins,
with detailed timing / progress logs.

Author: <your-name>
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# constants / helpers
# ──────────────────────────────────────────────────────────────────────────────
DTYPE_STR  = h5py.string_dtype(encoding="utf-8")
BUF_SIZE   = 1_000_000                      # records written per flush
RNG_SEED   = 42
MAX_LEN    = 512                            # skip proteins longer than this AA
SENTINELS  = {"prefix": "1", "middle": "2", "suffix": "3"}


def log(msg: str) -> None:
    """Write *msg* with ISO timestamp to stderr and flush immediately."""
    sys.stderr.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    sys.stderr.flush()


def parse_fasta(path: Path) -> Dict[str, str]:
    """Load a FASTA file into memory; return {accession → sequence}."""
    sequences: Dict[str, List[str]] = {}
    acc = None
    with path.open("rt") as fh:
        for line in fh:
            if line.startswith(">"):
                acc = line[1:].split()[0].strip()
                sequences[acc] = []
            else:
                sequences[acc].append(line.strip())
    return {k: "".join(v) for k, v in sequences.items()}


def fim_transform_full(seq: str, start: int, end: int) -> str:
    """Return full FIM representation: 1prefix3suffix2middle order"""
    prefix, middle, suffix = seq[:start], seq[start:end + 1], seq[end + 1:]
    full_fim = f"{SENTINELS['prefix']}{prefix}{SENTINELS['suffix']}{suffix}{SENTINELS['middle']}{middle}"
    return full_fim


def fim_transform_middle_only(seq: str, start: int, end: int) -> str:
    """Return middle-only FIM representation: 132middle order (prefix and suffix dropped)"""
    middle = seq[start:end + 1]
    middle_only_fim = f"{SENTINELS['prefix']}{SENTINELS['suffix']}{SENTINELS['middle']}{middle}"
    return middle_only_fim


def extend_dataset(ds, offset: int, values: List[str]) -> int:
    """Append *values* to dataset *ds* starting at *offset*; return new offset."""
    n_new = len(values)
    ds.resize(offset + n_new, axis=0)
    ds[offset:offset + n_new] = values
    return offset + n_new


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta",   required=True, type=Path, help="Big FASTA file")
    parser.add_argument("--h5_dir",  required=True, type=Path, help="Directory with part_*.h5 shards")
    parser.add_argument("--output",  default=Path("output_idrs.h5"), type=Path,
                        help="Output HDF5 filename")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"          # enforce single-threaded BLAS/HDF5

    total_start = time.perf_counter()

    # 1. FASTA -----------------------------------------------------------------
    log("Reading FASTA …")
    t0 = time.perf_counter()
    fasta = parse_fasta(args.fasta)
    targets = set(fasta.keys())
    log(f"FASTA loaded in {time.perf_counter() - t0:,.1f}s "
        f"({len(targets):,} distinct accessions)")

    # 2. Output file -----------------------------------------------------------
    log(f"Creating output file {args.output} …")
    with h5py.File(args.output, "w", libver="latest") as fout:
        ds = fout.create_dataset(
            "idrs",
            shape=(0,), maxshape=(None,),
            dtype=DTYPE_STR, chunks=(BUF_SIZE,),
        )

        write_buf: List[str] = []
        written       = 0        # total records written
        processed_ids = 0        # total IDR rows considered (after length check)

        # 3. Iterate through shards -------------------------------------------
        h5_files = sorted(p for p in args.h5_dir.glob("part_*.h5") if p.is_file())
        log(f"Discovered {len(h5_files)} HDF5 shards")

        for shard_no, shard in enumerate(h5_files, start=1):
            shard_t0 = time.perf_counter()
            log(f"[{shard_no}/{len(h5_files)}] Loading {shard.name} …")

            with h5py.File(shard, "r") as h5:
                acc_arr   = h5["accession_ids"][:]
                seq_arr   = h5["full_seq"][:]
                len_arr   = h5["full_length"][:]
                start_arr = h5["idr_start"][:]
                end_arr   = h5["idr_end"][:]

                rows_in_shard   = len(acc_arr)
                shard_written   = 0

                for idx, raw_id in enumerate(acc_arr):
                    # For each h5 shard, search for all accession IDs from the original FASTA 
                    # base UniProt accession (strip trailing “-…”)
                    base_acc = raw_id.decode("utf-8").split("-", 1)[0]
                    if base_acc not in targets:
                        continue
                    if len_arr[idx] > MAX_LEN:
                        continue

                    seq_full = seq_arr[idx].decode("utf-8")
                    start_pos = int(start_arr[idx])
                    end_pos = int(end_arr[idx])
                    
                    # Skip if both FIM transforms would be identical
                    # (happens when prefix and suffix are both empty)
                    if start_pos == 0 and end_pos == len(seq_full) - 1:
                        continue
                    
                    # Write full FIM (1prefix3suffix2middle)
                    full_fim = fim_transform_full(seq_full, start_pos, end_pos)
                    write_buf.append(full_fim)
                    processed_ids += 1
                    shard_written += 1
                    
                    # Write middle-only FIM (132middle)
                    middle_fim = fim_transform_middle_only(seq_full, start_pos, end_pos)
                    write_buf.append(middle_fim)
                    processed_ids += 1
                    shard_written += 1

                    if len(write_buf) >= BUF_SIZE:
                        written = extend_dataset(ds, written, write_buf)
                        log(f"  • Flushed {BUF_SIZE:,} records "
                            f"(total {written:,})")
                        write_buf.clear()

            # end shard file
            log(f"Completed {shard.name}: "
                f"{shard_written:,}/{rows_in_shard:,} IDRs kept "
                f"in {time.perf_counter() - shard_t0:,.1f}s")

        # after all shards -----------------------------------------------------
        if write_buf:
            written = extend_dataset(ds, written, write_buf)
            log(f"Final flush {len(write_buf):,} records "
                f"(grand total {written:,})")
            write_buf.clear()

    # total runtime -----------------------------------------------------------
    log(f"ALL DONE – {written:,} FIM sequences written "
        f"in {time.perf_counter() - total_start:,.1f}s "
        f"({processed_ids:,} IDR instances processed)")

if __name__ == "__main__":
    main()
