"""Extract residual-stream embeddings for downstream tasks.

Inputs may be a FASTA path, a single sequence string, or a list of sequences.

    uv run python examples/02_embeddings.py --model jxliu2/idiom-300M --layer 18
"""

import argparse

from idiom import IDiom

SEQS = [
    "MEDSKVDNRPQACDEFGHIKLMNPQRSTVWY",
    "GSGSQPQPQPGSGSGSNNNNQQQQGSGSGS",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="jxliu2/idiom-300M", help="HF repo id or local directory")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model = IDiom.from_pretrained(args.model, device=args.device)

    # One pooled vector per sequence -- the usual input to a downstream predictor.
    pooled, index = model.embed(SEQS, layers=[args.layer], pool="mean")[args.layer]
    print(f"pooled: {pooled.shape}  (one {pooled.shape[1]}-d vector per sequence)")
    print("  accessions:", [r["accession"] for r in index])

    # Per-residue rows, aligned 1:1 to residues (index carries the source position).
    per_res, index = model.embed(SEQS[0], layers=[args.layer], pool="none")[args.layer]
    print(f"per-residue: {per_res.shape}  for a {len(SEQS[0])}-residue sequence")
    print("  first row:", {k: index[0][k] for k in ("accession", "source_pos", "residue")})


if __name__ == "__main__":
    main()
