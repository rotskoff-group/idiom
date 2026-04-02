"""
Compute IDR perplexity for a set of FASTA datasets using the pre-trained IDR-PLM model.

Each dataset is sampled to 1000 sequences. Per-sequence perplexities are printed,
followed by per-dataset summary statistics.

Usage:
    python calculate_perplexity.py [--ckpt PATH] [--device cuda|cpu] [--n N] [--seed S]
"""

import argparse
import statistics
import sys
from pathlib import Path

from idr_plm.nn.transformer.module import LightningModel
from idr_plm.nn.transformer.utils.perplexity import compute_perplexity_from_fasta


# src/idr_plm/nn/transformer/utils/ -> repo root is 5 levels up
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT / "src"))

_DEFAULT_CKPT = str(
    _REPO_ROOT / "models/idr-plm/base/version_2/checkpoints/best_model_step_243022.ckpt"
)
_DEFAULT_SHARD = str(_REPO_ROOT / "models/data/shard/0001_file.h5")

# Datasets scored with IDR bounds from the header: 1{prefix}3{suffix}2{IDR}
DATASETS = {
    "generated_idps": "datasets/idr_datasets/generated_sequences/generated_idps/generated_full.fasta",
    "generated_idrs": "datasets/idr_datasets/generated_sequences/generated_idrs/generated_full.fasta",
    "generated_npm1": "datasets/idr_datasets/generated_sequences/generated_npm1/generated_full.fasta",
    "disprot_idrs": "datasets/idr_datasets/reference_sequences/DisProt/disprot_idrs.fasta",
    "training_afdb": "datasets/idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_full.fasta",
    "generated_chromosome": "datasets/idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_full.fasta",
    "generated_nucleolus": "datasets/idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_full.fasta",
    "generated_p-body": "datasets/idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_full.fasta",
    "generated_stress_granule": "datasets/idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_full.fasta",
}

# Datasets where the full sequence is treated as an IDP: FIM = "132{sequence}"
DATASETS_FULL_IDP = {
    "cath_s60": "datasets/idr_datasets/reference_sequences/CATH/cath-domain-seqs-S60_1000.fa",
}


def main():
    parser = argparse.ArgumentParser(
        description="Compute IDR perplexity from FASTA files."
    )
    parser.add_argument(
        "--ckpt", default=_DEFAULT_CKPT, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--shard",
        default=_DEFAULT_SHARD,
        help="Path to precomputed HDF5 shard (alphabet source)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Sequences to sample per dataset (default 1000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.ckpt}")
    print(f"Shard (alphabet):  {args.shard}")
    print(f"Device: {device}\n")

    model = LightningModel.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model.to(device)

    all_datasets = [(name, rel_path, False) for name, rel_path in DATASETS.items()]
    all_datasets += [
        (name, rel_path, True) for name, rel_path in DATASETS_FULL_IDP.items()
    ]

    for name, rel_path, full_idp in all_datasets:
        fasta_path = _REPO_ROOT / rel_path
        if not fasta_path.exists():
            print(f"[{name}] SKIPPED — file not found: {fasta_path}\n")
            continue

        print(f"{'=' * 60}")
        print(f"Dataset: {name}")
        print(f"File:    {fasta_path}")
        if full_idp:
            print("Mode:    full-sequence IDP (FIM = 132{sequence})")

        results = compute_perplexity_from_fasta(
            model,
            str(fasta_path),
            shard_path=args.shard,
            n_sample=args.n,
            seed=args.seed,
            device=device,
            full_sequence_as_idp=full_idp,
        )

        if not results:
            print("  No valid sequences found.\n")
            continue

        ppl_idrs = [ppl_idr for _, ppl_idr, _ in results]
        ppl_fulls = [ppl_full for _, _, ppl_full in results]

        print(f"Sequences scored:          {len(results)}")
        print(
            f"  IDR-only  — mean: {statistics.mean(ppl_idrs):.4f}  median: {statistics.median(ppl_idrs):.4f}  stdev: {statistics.stdev(ppl_idrs):.4f}"
        )
        print(
            f"  Full seq  — mean: {statistics.mean(ppl_fulls):.4f}  median: {statistics.median(ppl_fulls):.4f}  stdev: {statistics.stdev(ppl_fulls):.4f}"
        )
        print()


if __name__ == "__main__":
    main()
