"""
CLI for creating GRPO RL training datasets.

Subcommands:
  idp   -- Generate generic IDP RL dataset (repeated "132" prompts)
  idr   -- Generate IDR RL dataset from a FASTA file (fixed output name)
"""

import argparse
import os
import pickle

import h5py
import numpy as np

from idiom.scripts.data.make_infer_prompt import (
    SENTINELS,
    fim_transform,
    load_alphabet,
    parse_fasta,
    tokenize_prompts,
)

NUM_DUPLICATES = 1000


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
    output_h5 = os.path.join(out_dir, f"{name}_dataset.h5")

    with h5py.File(output_h5, "w") as dst:
        dst.create_dataset("tokens", data=tokens, dtype=np.int32)
        dst.create_dataset("masks", data=masks, dtype=np.int32)
        src.copy("alphabet", dst)
        src.copy("input_metadata", dst)

    src.close()
    print(f"Wrote GRPO dataset to {output_h5}")


def cmd_idp(args):
    alphabet = load_alphabet(args.shard)
    name = args.name

    prompts = ["132"] * NUM_DUPLICATES
    metadata_list = [("idp", 0, None, None, None, None, p, p) for p in prompts]
    prompt_array = tokenize_prompts(prompts, alphabet)

    os.makedirs(args.out_dir, exist_ok=True)

    array_pkl_path = os.path.join(args.out_dir, f"{name}_array.pkl")
    metadata_pkl_path = os.path.join(args.out_dir, f"{name}_metadata.pkl")

    with open(array_pkl_path, "wb") as f:
        pickle.dump(prompt_array, f)

    with open(metadata_pkl_path, "wb") as f:
        pickle.dump({"prompts": prompts, "metadata_list": metadata_list}, f)

    print(f"Saved prompts to {args.out_dir}")
    print(f"  Array file:    {os.path.basename(array_pkl_path)}")
    print(f"  Metadata file: {os.path.basename(metadata_pkl_path)}")

    make_grpo_dataset(array_pkl_path, args.shard, args.out_dir, name)


def cmd_idr(args):
    alphabet = load_alphabet(args.shard)
    name = args.name

    sequences = list(parse_fasta(args.fasta))
    if len(sequences) != 1:
        raise ValueError(
            f"Expected exactly 1 sequence in {args.fasta}, found {len(sequences)}"
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

    os.makedirs(args.out_dir, exist_ok=True)

    array_pkl_path = os.path.join(args.out_dir, f"{name}_array.pkl")
    metadata_pkl_path = os.path.join(args.out_dir, f"{name}_metadata.pkl")

    with open(array_pkl_path, "wb") as f:
        pickle.dump(prompt_array, f)

    with open(metadata_pkl_path, "wb") as f:
        pickle.dump({"prompts": prompts, "metadata_list": metadata_list}, f)

    print(f"Saved prompts to {args.out_dir}")
    print(f"  Array file:    {os.path.basename(array_pkl_path)}")
    print(f"  Metadata file: {os.path.basename(metadata_pkl_path)}")

    make_grpo_dataset(array_pkl_path, args.shard, args.out_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Create a GRPO RL training dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # idp subcommand
    idp_parser = subparsers.add_parser(
        "idp", help="Generate generic IDP RL dataset (repeated '132' prompts)"
    )
    idp_parser.add_argument(
        "--name",
        type=str,
        default="idp_prompt_grpo",
        help="Output file base name (default: idp_prompt_grpo)",
    )
    idp_parser.add_argument(
        "--shard",
        type=str,
        default="models/data/shard/0001_file.h5",
        help="Path to the precomputed shard .h5 file (default: models/data/shard/0001_file.h5)",
    )
    idp_parser.add_argument(
        "--out_dir",
        type=str,
        default="models/data/rl_datasets",
        help="Output directory (default: models/data/rl_datasets)",
    )

    # idr subcommand
    idr_parser = subparsers.add_parser(
        "idr", help="Generate IDR RL dataset from a FASTA file (fixed output name)"
    )
    idr_parser.add_argument(
        "--name",
        type=str,
        default="idr_prompt_grpo",
        help="Output file base name (default: idr_prompt_grpo)",
    )
    idr_parser.add_argument(
        "--fasta", type=str, required=True, help="Path to input FASTA file"
    )
    idr_parser.add_argument(
        "--shard",
        type=str,
        default="models/data/shard/0001_file.h5",
        help="Path to the precomputed shard .h5 file (default: models/data/shard/0001_file.h5)",
    )
    idr_parser.add_argument(
        "--out_dir",
        type=str,
        default="models/data/rl_datasets",
        help="Output directory (default: models/data/rl_datasets)",
    )

    args = parser.parse_args()

    if args.command == "idp":
        cmd_idp(args)
    elif args.command == "idr":
        cmd_idr(args)


if __name__ == "__main__":
    main()
