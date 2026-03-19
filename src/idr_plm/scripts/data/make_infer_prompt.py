"""
CLI for creating inference prompts.

Subcommands:
  idp       -- Generate IDP prompts (repeated "132")
  specific  -- Generate FIM prompts from a FASTA file
"""

import argparse
import os
import pickle

import h5py
import numpy as np

from idr_plm.nn.transformer.utils.tokenizer import CharTokenizer

SENTINELS = {"prefix": "1", "middle": "2", "suffix": "3"}


def load_alphabet(shard):
    precomputed_shard = h5py.File(shard, "r")
    alphabet = precomputed_shard["alphabet"][:]
    return [x.decode("utf-8") for x in alphabet]


def tokenize_prompts(prompts, alphabet):
    tokenizer = CharTokenizer()
    arrays = []
    for p in prompts:
        tokens = tokenizer.tokenize(p)
        indices = [alphabet.index(x) for x in tokens]
        arrays.append(np.array(indices, dtype=np.int32))
    return arrays


def save_outputs(name, prompt_array, prompts, metadata_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    array_filename = f"{name}_array.pkl"
    metadata_filename = f"{name}_metadata.pkl"

    with open(os.path.join(out_dir, array_filename), "wb") as f:
        pickle.dump(prompt_array, f)

    with open(os.path.join(out_dir, metadata_filename), "wb") as f:
        pickle.dump({"prompts": prompts, "metadata_list": metadata_list}, f)

    print(f"Saved to {out_dir}")
    print(f"  Array file:    {array_filename}")
    print(f"  Metadata file: {metadata_filename}")


def fim_transform(seq: str, start: int, end: int) -> str:
    prefix = seq[:start]
    middle = seq[start : end + 1]
    suffix = seq[end + 1 :]
    return f"{SENTINELS['prefix']}{prefix}{SENTINELS['suffix']}{suffix}{SENTINELS['middle']}{middle}"


def parse_fasta(fasta_path):
    """Yield (header, sequence) pairs from a FASTA file."""
    header = None
    seq_lines = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        yield header, "".join(seq_lines)


def cmd_idp(args):
    alphabet = load_alphabet(args.shard)
    num_duplicates = args.num_duplicates
    name = f"idp_prompt_{num_duplicates}x"

    prompts = ["132"] * num_duplicates
    metadata_list = [("idp", 0, None, None, None, None, p, p) for p in prompts]
    prompt_array = tokenize_prompts(prompts, alphabet)

    save_outputs(name, prompt_array, prompts, metadata_list, args.out_dir)


def cmd_specific(args):
    alphabet = load_alphabet(args.shard)
    num_duplicates = args.num_duplicates
    fasta_stem = os.path.splitext(os.path.basename(args.fasta))[0]

    all_prompts = []
    all_metadata = []

    for header, og_sequence in parse_fasta(args.fasta):
        # Parse header: must contain _IDR_x-y
        if "_IDR_" not in header:
            print(f"Skipping {header!r}: no _IDR_ in header")
            continue

        acc_part, idr_part = header.split("_IDR_", 1)
        acc = acc_part.strip()
        try:
            idr_start_str, idr_end_str = idr_part.split("-", 1)
            idr_start = int(idr_start_str)
            idr_end = int(idr_end_str)
        except ValueError:
            print(f"Skipping {header!r}: cannot parse IDR range from {idr_part!r}")
            continue

        # 1-indexed: convert to 0-indexed for slicing
        idr_seq = og_sequence[idr_start - 1 : idr_end]
        fim_seq = fim_transform(og_sequence, idr_start - 1, idr_end - 1)

        mid_idx = fim_seq.find(SENTINELS["middle"])
        fim_prompt = fim_seq[: mid_idx + 1] if mid_idx != -1 else fim_seq

        if fim_prompt in ["132", "312"]:
            print(f"Skipping {acc}: degenerate prompt ({fim_prompt!r})")
            continue

        protein_data = (
            acc,
            0,
            idr_start,
            idr_end,
            idr_seq,
            og_sequence,
            fim_seq,
            fim_prompt,
        )
        all_prompts.extend([fim_prompt] * num_duplicates)
        all_metadata.extend([protein_data] * num_duplicates)

    prompt_array = tokenize_prompts(all_prompts, alphabet)
    name = f"{fasta_stem}_prompt_{num_duplicates}x"
    save_outputs(name, prompt_array, all_prompts, all_metadata, args.out_dir)


def main():
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))

    parser = argparse.ArgumentParser(
        description="Make inference prompts for the transformer."
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=os.path.join(_repo_root, "models/data/shard/0001_file.h5"),
        help="Path to the precomputed shard .h5 file (default: models/data/shard/0001_file.h5)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_repo_root, "models/data/prompts"),
        help="Output directory for prompt files (default: models/data/prompts)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # idp subcommand
    idp_parser = subparsers.add_parser(
        "idp", help="Generate IDP prompts (repeated '132')"
    )
    idp_parser.add_argument(
        "--num_duplicates", type=int, required=True, help="Number of '132' prompts"
    )

    # specific subcommand
    specific_parser = subparsers.add_parser(
        "specific", help="Generate FIM prompts from a FASTA file"
    )
    specific_parser.add_argument(
        "--fasta", type=str, required=True, help="Path to input FASTA file"
    )
    specific_parser.add_argument(
        "--num_duplicates",
        type=int,
        default=100000,
        help="Duplicates per protein (default: 100000)",
    )

    args = parser.parse_args()

    if args.command == "idp":
        cmd_idp(args)
    elif args.command == "specific":
        cmd_specific(args)


if __name__ == "__main__":
    main()
