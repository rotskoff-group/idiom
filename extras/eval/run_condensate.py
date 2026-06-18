"""``python -m extras.eval.run_condensate`` — validation harness for ProtGPS-reward GRPO runs.

Generates (or loads) de-novo IDPs from a base (pre-RL) checkpoint and one or more GRPO checkpoints,
then scores them along three axes and writes a canonical results layout consumed by the
``extras.figures.condensate_rl`` figures:

  - ProtGPS specificity (in-distribution): each model x 12 compartments + target-selectivity matrix.
  - DeepLoc localization (orthogonal): subcellular-compartment probabilities.
  - catGRANULE 2.0 LLPS (orthogonal): phase-separation propensity.
  - Naturalness (reward-hacking control): sparrow features + AA entropy + LCD fraction + de-novo
    perplexity under the base model, vs a natural-IDR reference.

    python -m extras.eval.run_condensate \\
        --base-ckpt .../01_pretrain/.../epoch_4_step_234428.ckpt \\
        --grpo-ckpt chromosome=.../02_grpo/chromosome/checkpoints/last.ckpt \\
        --grpo-ckpt nucleolus=.../02_grpo/nucleolus/checkpoints/last.ckpt \\
        --n 500 --max-new-tokens 1000 --natural-ref AFDB_test.fasta \\
        --tools protgps,deeploc,llps,naturalness --out .../04_condensate
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

# ProtGPS condensate target -> expected DeepLoc organelle class (None = control / no expectation).
EXPECTED_ORGANELLE = {
    "base": None, "chromosome": "Nucleus", "nucleolus": "Nucleus",
    "p-body": "Cytoplasm", "stress_granule": "Cytoplasm",
}


def _write_fasta(seqs, path, prefix):
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{prefix}_{i}\n{s}\n")


def _denovo_records_fasta(seqs, path, prefix):
    """Each IDR as its own de-novo record (``_IDR_1-len``, full=IDR) for de-novo perplexity."""
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{prefix}_{i}_IDR_1-{len(s)}\n{s}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-ckpt", help="pre-RL checkpoint (label 'base'; control)")
    p.add_argument("--grpo-ckpt", action="append", default=[], metavar="name=path",
                   help="GRPO checkpoint as label=path (repeatable)")
    p.add_argument("--generations-dir", help="dir of <label>.fasta to score instead of generating")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--natural-ref", help="natural-IDR record FASTA (for the naturalness comparison)")
    p.add_argument("--n-natural", type=int, default=2000)
    p.add_argument("--tools", default="protgps,deeploc,llps,naturalness",
                   help="comma list: protgps,deeploc,llps,naturalness")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    out = Path(args.out)
    (out / "fastas").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    from idiom import IDiom
    from idiom.data.io import read_fasta

    # 1. collect per-model sequences (generate or load)
    models: dict[str, str] = {}
    if args.base_ckpt:
        models["base"] = args.base_ckpt
    for spec in args.grpo_ckpt:
        label, path = spec.split("=", 1)
        models[label] = path

    seqs_by: dict[str, list[str]] = {}
    if args.generations_dir:
        for f in sorted(Path(args.generations_dir).glob("*.fasta")):
            seqs_by[f.stem] = [s for _, s in read_fasta(f) if s]
    else:
        for label, ckpt in models.items():
            print(f"[gen] {label} <- {ckpt}", flush=True)
            m = IDiom.load(ckpt, device="auto")
            seqs = [s for s in m.generate_idp(n=args.n, max_new_tokens=args.max_new_tokens,
                                              temperature=args.temperature, seed=args.seed) if s]
            seqs_by[label] = seqs
            _write_fasta(seqs, out / "fastas" / f"{label}.fasta", label)
            del m
    labels = list(seqs_by)
    summary: dict = {"n": {k: len(v) for k, v in seqs_by.items()}}

    # 2. ProtGPS specificity (in-distribution)
    if "protgps" in tools:
        from extras.eval.protgps import get_compartments, specificity_matrix
        from extras.eval.protgps import protgps_score
        comps = get_compartments()
        (out / "scores" / "protgps").mkdir(exist_ok=True)
        scores_by = {}
        for label in labels:
            mat = protgps_score(seqs_by[label])
            scores_by[label] = mat
            with (out / "scores" / "protgps" / f"{label}.csv").open("w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["idx", *comps])
                for i, row in enumerate(mat):
                    w.writerow([i, *[f"{x:.4f}" for x in row]])
            print(f"[protgps] {label}: n={len(seqs_by[label])}", flush=True)
        summary["protgps_specificity"] = specificity_matrix(scores_by)

    # 3. DeepLoc localization (orthogonal)
    if "deeploc" in tools:
        from extras.eval.localization import COMPARTMENTS, deeploc_score, localization_stats
        (out / "scores" / "deeploc").mkdir(exist_ok=True)
        summary["deeploc"] = {}
        for label in labels:
            sc = deeploc_score(seqs_by[label])
            with (out / "scores" / "deeploc" / f"{label}.csv").open("w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["idx", *COMPARTMENTS])
                for i in range(len(seqs_by[label])):
                    w.writerow([i, *[f"{sc[c][i]:.4f}" for c in COMPARTMENTS]])
            summary["deeploc"][label] = localization_stats(sc, expected=EXPECTED_ORGANELLE.get(label))
            print(f"[deeploc] {label}", flush=True)

    # 4. catGRANULE LLPS (orthogonal)
    if "llps" in tools:
        from extras.eval.llps import catgranule_score, llps_stats
        (out / "scores" / "llps").mkdir(exist_ok=True)
        summary["llps"] = {}
        for label in labels:
            sc = catgranule_score(seqs_by[label])
            with (out / "scores" / "llps" / f"{label}.csv").open("w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["idx", "LLPS_Score"])
                for i, v in enumerate(sc):
                    w.writerow([i, f"{v:.4f}"])
            summary["llps"][label] = llps_stats(sc)
            print(f"[llps] {label}", flush=True)

    # 5. naturalness (reward-hacking control)
    if "naturalness" in tools:
        from extras.eval.metrics import FEATURES, composition_extras_table, features_table, summarize
        from idiom.data.io import read_records
        from idiom.utils.perplexity import perplexity
        nat = dict(seqs_by)
        if args.natural_ref:
            rng = random.Random(args.seed)
            recs = list(read_records(args.natural_ref)); rng.shuffle(recs)
            nat["natural_ref"] = [r.full_seq[r.idr_start:r.idr_end] for r in recs[: args.n_natural]
                                  if r.full_seq[r.idr_start:r.idr_end]]
        base_model = IDiom.load(args.base_ckpt, device="auto") if args.base_ckpt else None
        rows = []
        tmp = out / "scores" / "_ppl.fasta"
        for label, seqs in nat.items():
            feats = summarize({**features_table(seqs), **composition_extras_table(seqs)})
            row = {"model": label, "n": len(seqs)}
            for f in (*FEATURES, "aa_entropy", "lcd_fraction"):
                row[f] = feats[f]["mean"]
            if base_model is not None:
                _denovo_records_fasta(seqs, tmp, label)
                ppl = perplexity(base_model.model, str(tmp), fim_full_prob=0.0,
                                 device=str(base_model.device), tokenizer=base_model.tok,
                                 batch_size=64, max_records=len(seqs))
                row["base_nll"], row["base_ppl"] = ppl["nll"], ppl["perplexity"]
            rows.append(row); print(f"[naturalness] {label}", flush=True)
        if tmp.exists():
            tmp.unlink()
        with (out / "scores" / "naturalness.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary["naturalness"] = rows

    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n-> {out/'metrics.json'}")


if __name__ == "__main__":
    main()
