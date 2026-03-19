#!/usr/bin/env python3
"""
Randomly split the dataset  ['residues']  of an input HDF5 file into N part_X_residues.h5 files.

Example
-------
$ python split_parts.py models/data/full_residues.h5 \
      --num-parts 500 \
      --out-dir ./splits \
      --chunk-size 1_000_000 \
      --seed 42
"""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Randomly partition the 'residues' dataset of an HDF5 file."
    )
    p.add_argument(
        "input", help="Path to the source .h5 file containing dataset ['residues']."
    )
    p.add_argument(
        "-n",
        "--num-parts",
        type=int,
        required=True,
        help="Number of partition files to produce (X in part_X_residues.h5).",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default=".",
        help="Directory in which to create the part_X_residues.h5 files. "
        "It is created if it does not yet exist.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of records read at once (default 1 000 000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible splits.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Probe the source file just once to get length and dtype ------------------
    with h5py.File(args.input, "r") as fin:
        src = fin["residues"]
        total = int(src.shape[0])
        idrs_dtype = src.dtype  # fixed-length or vlen byte strings

    # -------------------------------------------------------------------------
    # Prepare output files with extendable datasets (shape=(0,) but maxshape=None)
    part_files = []
    part_dsets = []
    for i in range(1, args.num_parts + 1):
        fout = h5py.File(os.path.join(args.out_dir, f"part_{i}_residues.h5"), "w")
        dset = fout.create_dataset(
            "residues",
            shape=(0,),
            maxshape=(None,),
            dtype=idrs_dtype,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
        part_files.append(fout)
        part_dsets.append(dset)

    # -------------------------------------------------------------------------
    # Stream the source dataset and distribute records -------------------------
    with h5py.File(args.input, "r") as fin:
        src = fin["residues"]
        for start in tqdm(
            range(0, total, args.chunk_size),
            total=(total + args.chunk_size - 1) // args.chunk_size,
            unit="chunk",
            desc="Splitting",
        ):
            end = min(start + args.chunk_size, total)
            block = src[start:end]  # (block_size,)
            # Draw a random part index for every record in the block
            targets = rng.integers(0, args.num_parts, size=len(block))

            # Append to each part --------------------------
            for idx in range(args.num_parts):
                mask = targets == idx
                if not np.any(mask):
                    continue
                subset = block[mask]
                dset = part_dsets[idx]
                old_n = dset.shape[0]
                dset.resize(old_n + len(subset), axis=0)
                dset[old_n:] = subset

    for f in part_files:
        f.close()

    print("Completed: wrote", args.num_parts, "partition files to", args.out_dir)


if __name__ == "__main__":
    main()
