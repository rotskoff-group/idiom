#!/usr/bin/env python3
"""
Prepare HDF5 part files for transformer_precompute.

Steps
-----
1. For every *_idrs.h5 / *_residues.h5 file in the working directory,
   create a matching *_targs.h5 with a zero-filled 'targets' dataset
   (make_targs logic).
2. Split the 'residues' dataset of a source HDF5 file into N part files
   named part_X_residues.h5, then create matching *_targs.h5 files for
   each part (split_parts logic).

Example
-------
$ make_precompute_parts models/data/full_residues.h5 \\
      --num-parts 500 \\
      --out-dir ./splits \\
      --chunk-size 1_000_000 \\
      --seed 42
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Split a 'residues' HDF5 dataset into N parts and create "
            "zero-filled targets files for each part."
        )
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
        help="Directory in which to create the part files. Created if absent.",
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


def make_targs(directory: str) -> None:
    """Create *_targs.h5 files for every *_idrs.h5 and *_residues.h5 in directory."""
    for suffix, dataset_key in [("_idrs.h5", "idrs"), ("_residues.h5", "residues")]:
        for src_filename in os.listdir(directory):
            if not src_filename.endswith(suffix):
                continue
            src_path = os.path.join(directory, src_filename)
            with h5py.File(src_path, "r") as src_file:
                if dataset_key not in src_file:
                    print(
                        f"'{dataset_key}' dataset not found in {src_filename}, skipping."
                    )
                    continue
                length = len(src_file[dataset_key])

            targs_filename = src_filename.replace(suffix, "_targs.h5")
            targs_path = os.path.join(directory, targs_filename)
            with h5py.File(targs_path, "w") as targs_file:
                targs_file.create_dataset(
                    "targets", data=np.zeros(length, dtype="float32")
                )

            print(f"Created {targs_path} with {length} zeros.")


def split_parts(
    input_path: str,
    num_parts: int,
    out_dir: str,
    chunk_size: int,
    seed: int | None,
) -> None:
    """Randomly partition the 'residues' dataset of input_path into num_parts files."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    with h5py.File(input_path, "r") as fin:
        src = fin["residues"]
        total = int(src.shape[0])
        idrs_dtype = src.dtype

    part_files = []
    part_dsets = []
    for i in range(1, num_parts + 1):
        fout = h5py.File(os.path.join(out_dir, f"part_{i}_residues.h5"), "w")
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

    with h5py.File(input_path, "r") as fin:
        src = fin["residues"]
        for start in tqdm(
            range(0, total, chunk_size),
            total=(total + chunk_size - 1) // chunk_size,
            unit="chunk",
            desc="Splitting",
        ):
            end = min(start + chunk_size, total)
            block = src[start:end]
            targets = rng.integers(0, num_parts, size=len(block))

            for idx in range(num_parts):
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

    print(f"Completed: wrote {num_parts} partition files to {out_dir}")


def main() -> None:
    args = parse_args()

    # Step 1: split the source file into parts
    split_parts(
        input_path=args.input,
        num_parts=args.num_parts,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    # Step 2: create zero-filled targets files for each part and make the shard dir
    make_targs(args.out_dir)
    precompute_shards = os.path.join(args.out_dir, "precompute_shards")
    os.makedirs(precompute_shards, exist_ok=True)
    print(f"Created {precompute_shards}/")


if __name__ == "__main__":
    main()
