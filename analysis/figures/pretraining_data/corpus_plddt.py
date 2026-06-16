"""SI figure: pLDDT distributions of the final training corpus (sampled).

Left: mean pLDDT over each (final) IDR span. Right: the max per-residue pLDDT over each full protein.
Per-residue pLDDT is joined from the extraction master h5 (`final_corpus.plddt_by_protein`) to the
final-corpus records; thresholds (disordered < 70, folded > 80) are marked. Because the curation
already removed fully-low-pLDDT proteins, essentially every surviving protein has a confident folded
segment (max pLDDT >= 80) — the right panel confirms the filter on the actual training set.

    python analysis/figures/pretraining_data/corpus_plddt.py --n 200000
"""

from __future__ import annotations

import argparse

import numpy as np

from analysis.figures._style import COLORS, row_fig, save_fig, use_style
from analysis.figures.pretraining_data.final_corpus import FINAL_FASTA, MASTER_H5, plddt_by_protein, sample_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=FINAL_FASTA)
    ap.add_argument("--h5", default=MASTER_H5)
    ap.add_argument("--n", type=int, default=200_000, help="records to sample (contiguous)")
    ap.add_argument("--name", default="corpus_plddt")
    args = ap.parse_args()

    use_style()

    recs = sample_records(args.fasta, args.n)
    pl = plddt_by_protein([r[0] for r in recs], args.h5)
    mean_idr = np.array([pl[a][s:e].mean() for a, _seq, s, e in recs if a in pl], dtype=np.float32)
    max_full = np.array([p.max() for p in pl.values()], dtype=np.float32)  # one per unique protein

    fig, (ax1, ax2) = row_fig(2)

    ax1.hist(mean_idr, bins=80, range=(0, 100), color=COLORS["darkblue"])
    ax1.axvline(70, color=COLORS["red"], lw=1.5, ls="--")
    ax1.set_xlabel("mean IDR pLDDT")
    ax1.set_ylabel("count")
    ax1.set_title(f"IDRs (n = {len(mean_idr):,})")

    ax2.hist(max_full, bins=80, range=(0, 100), color=COLORS["lightblue"])
    ax2.axvline(80, color=COLORS["red"], lw=1.5, ls="--")
    frac = 100 * np.mean(max_full < 80)
    ax2.set_xlabel("max protein pLDDT")
    ax2.set_ylabel("count")
    ax2.set_title(f"proteins (n = {len(max_full):,}; max < 80: {frac:.1f}%)")

    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
