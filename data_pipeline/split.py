"""Random train/val/test split of a record FASTA (curation stage 5).

A plain random partition over IDR *records* (one entry = one `_IDR_x-y` IDR). The input is the
DisProt-deduped record FASTA; outputs `train/val/test.fasta` consumed by
`idiom.data.datamodule.RecordDataModule`. Pure split is CPU-testable; the CLI is the operator step.
"""

from __future__ import annotations

import random
from pathlib import Path


def _iter_fasta(path: str | Path):
    header, parts = None, []
    with Path(path).open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line, []
            else:
                parts.append(line)
        if header is not None:
            yield header, "".join(parts)


def split_records(
    records: list[tuple[str, str]],
    fractions: tuple[float, float, float] = (0.99, 0.005, 0.005),
    *,
    seed: int = 0,
) -> tuple[list, list, list]:
    """Shuffle and partition `(header, seq)` records into (train, val, test). Test gets the rest."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n = len(records)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    cut = [idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]]
    return tuple([records[i] for i in part] for part in cut)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Stage 5: random train/val/test split of a record FASTA.")
    ap.add_argument("--fasta", required=True, help="DisProt-deduped record FASTA")
    ap.add_argument("--out-dir", required=True, help="dir for train/val/test.fasta")
    ap.add_argument("--fractions", type=float, nargs=3, default=(0.99, 0.005, 0.005))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = list(_iter_fasta(args.fasta))
    train, val, test = split_records(records, tuple(args.fractions), seed=args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("val", val), ("test", test)):
        with (out / f"{name}.fasta").open("w") as fh:
            for header, seq in recs:
                fh.write(f"{header}\n{seq}\n")
        print(f"{name}: {len(recs):,} records")


if __name__ == "__main__":
    main()
