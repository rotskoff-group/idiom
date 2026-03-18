"""
CLI for creating GRPO RL training datasets from a FASTA file.

Usage:
    make_rl_dataset --fasta <path> [--shard <path>] [--out_dir <path>]
"""

import argparse
import os
import pickle

import h5py
import numpy as np

from idr_plm.scripts.data.make_infer_prompt import (
    SENTINELS,
    fim_transform,
    load_alphabet,
    parse_fasta,
    tokenize_prompts,
)

NUM_DUPLICATES = 1000


def make_prompts(fasta_path, shard, out_dir):
    alphabet = load_alphabet(shard)
    fasta_stem = os.path.splitext(os.path.basename(fasta_path))[0]

    sequences = list(parse_fasta(fasta_path))
    if len(sequences) != 1:
        raise ValueError(
            f"Expected exactly 1 sequence in {fasta_path}, found {len(sequences)}"
        )

    header, og_sequence = sequences[0]

    if "_IDR_" not in header:
        raise ValueError(f"No _IDR_ in header: {header!r}")

    acc_part, idr_part = header.split("_IDR_", 1)
    acc = acc_part.strip()
    try:
        idr_start_str, idr_end_str = idr_part.split("-", 1)
        idr_start = int(idr_start_str)
        idr_end = int(idr_end_str)
    except ValueError:
        raise ValueError(
            f"Cannot parse IDR range from {idr_part!r} in header {header!r}"
        )

    # 1-indexed: convert to 0-indexed for slicing
    idr_seq = og_sequence[idr_start - 1 : idr_end]
    fim_seq = fim_transform(og_sequence, idr_start - 1, idr_end - 1)

    mid_idx = fim_seq.find(SENTINELS["middle"])
    fim_prompt = fim_seq[: mid_idx + 1] if mid_idx != -1 else fim_seq

    if fim_prompt in ["132", "312"]:
        raise ValueError(f"Degenerate prompt for {acc}: {fim_prompt!r}")

    prompts = [fim_prompt] * NUM_DUPLICATES
    metadata_list = [
        (acc, 0, idr_start, idr_end, idr_seq, og_sequence, fim_seq, fim_prompt)
    ] * NUM_DUPLICATES

    prompt_array = tokenize_prompts(prompts, alphabet)

    name = f"{fasta_stem}_prompt_{NUM_DUPLICATES}x"
    os.makedirs(out_dir, exist_ok=True)

    array_pkl_path = os.path.join(out_dir, f"{name}_array.pkl")
    metadata_pkl_path = os.path.join(out_dir, f"{name}_metadata.pkl")

    with open(array_pkl_path, "wb") as f:
        pickle.dump(prompt_array, f)

    with open(metadata_pkl_path, "wb") as f:
        pickle.dump({"prompts": prompts, "metadata_list": metadata_list}, f)

    print(f"Saved prompts to {out_dir}")
    print(f"  Array file:    {os.path.basename(array_pkl_path)}")
    print(f"  Metadata file: {os.path.basename(metadata_pkl_path)}")

    return array_pkl_path, name


def make_grpo_dataset(array_pkl, shard, out_dir, name):
    with open(array_pkl, "rb") as f:
        prompts = pickle.load(f)

    src = h5py.File(shard, "r")
    pad_token = src["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]

    maxlen = max(len(arr) for arr in prompts)
    tokens = np.stack(
        [
            np.pad(arr, (0, maxlen - len(arr)), constant_values=pad_token)
            for arr in prompts
        ],
        axis=0,
    )
    masks = (tokens != pad_token).astype(np.int32)

    os.makedirs(out_dir, exist_ok=True)
    output_h5 = os.path.join(out_dir, f"{name}_grpo_dataset.h5")

    with h5py.File(output_h5, "w") as dst:
        dst.create_dataset("tokens", data=tokens, dtype=np.int32)
        dst.create_dataset("masks", data=masks, dtype=np.int32)
        src.copy("alphabet", dst)
        src.copy("input_metadata", dst)

    src.close()
    print(f"Wrote GRPO dataset to {output_h5}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a GRPO RL training dataset from a FASTA file."
    )
    parser.add_argument(
        "--fasta", type=str, required=True, help="Path to input FASTA file"
    )
    parser.add_argument(
        "--shard",
        type=str,
        default="models/data/shard/0001_file.h5",
        help="Path to the precomputed shard .h5 file (default: models/data/shard/0001_file.h5)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="models/data/rl_datasets/specific_dataset",
        help="Output directory (default: models/data/rl_datasets/specific_dataset)",
    )
    args = parser.parse_args()

    array_pkl, name = make_prompts(args.fasta, args.shard, args.out_dir)
    make_grpo_dataset(array_pkl, args.shard, args.out_dir, name)


if __name__ == "__main__":
    main()
