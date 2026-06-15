"""SI figure: IDR and full-protein length distributions of the extracted AFDB corpus, vs DisProt.

Corpus lengths come from the merged extraction h5 (one row per IDR). Overlaid is DisProt under the
*same* filter used in `composition_vs_disprot.py` (`disprot_regions`: non-IDP, IDR >= 30, full seq
<= 1020) — one entry per qualifying 'D' region, matching the corpus' per-IDR-record convention.
Distributions are density-normalized because DisProt is ~10^3x smaller than the corpus. Marks the
512 / 1024 context budgets on the protein-length panel (<= max_len - 4 is what v2 keeps).

    python analysis/figures/pretraining_data/corpus_lengths.py --h5 /path/AFDB_IDR_90_alldata.h5
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from analysis.figures._style import COLORS, row_fig, save_fig, use_style
from analysis.figures.pretraining_data.composition_vs_disprot import disprot_regions

DATA = "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data"
DEFAULT_H5 = f"{DATA}/pretraining/AFDB/clustering_90/AFDB_IDR_90_alldata.h5"
DEFAULT_DISPROT = f"{DATA}/reference/disprot/DisProt release_2025_06 with_ambiguous_evidences.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--disprot-json", default=DEFAULT_DISPROT)
    ap.add_argument("--name", default="corpus_length_distributions")
    args = ap.parse_args()

    use_style()

    with h5py.File(args.h5, "r") as f:
        idr = f["idr_length"][:].astype(np.int32)
        full = f["full_length"][:].astype(np.int32)

    dp = list(disprot_regions(args.disprot_json))
    dp_idr = np.array([len(s) for s, _ in dp], dtype=np.int32)
    dp_full = np.array([n for _, n in dp], dtype=np.int32)

    fig, (ax1, ax2) = row_fig(2)
    hk = dict(density=True, bins=120)
    sk = dict(density=True, bins=60, histtype="step", lw=2.0, color=COLORS["green"])

    ax1.hist(idr, range=(0, 600), color=COLORS["darkblue"], label="corpus", **hk)
    ax1.hist(dp_idr, range=(0, 600), label="DisProt", **sk)
    ax1.set_xlabel("IDR length (residues)")
    ax1.set_ylabel("density")
    ax1.set_title(f"IDRs (corpus n = {len(idr):,})")
    ax1.legend(frameon=False)

    ax2.hist(full, range=(0, 1500), color=COLORS["lightblue"], label="corpus", **hk)
    ax2.hist(dp_full, range=(0, 1500), label="DisProt", **sk)
    for cut in (512, 1024):
        ax2.axvline(cut, color=COLORS["red"], lw=1.5, ls="--")
        ax2.text(cut + 10, ax2.get_ylim()[1] * 0.92, str(cut), color=COLORS["red"], fontsize=14)
    kept = 100 * np.mean(full <= 1020)
    ax2.set_xlabel("protein length (residues)")
    ax2.set_ylabel("density")
    ax2.set_title(f"proteins (corpus ≤1020: {kept:.0f}%)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    out = save_fig(fig, args.name, subdir="si_figs/dataset")
    print("saved", out, f"| DisProt regions: {len(dp):,}")


if __name__ == "__main__":
    main()
