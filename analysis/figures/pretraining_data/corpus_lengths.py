"""SI figure: IDR and full-protein length distributions of the extracted AFDB corpus.

Reads the length fields from the merged extraction h5 (one row per IDR). Marks the 512 / 1024
context budgets on the protein-length panel (≤ max_len − 4 is what v2 keeps).

    python analysis/figures/corpus_lengths.py --h5 /path/AFDB_IDR_90_alldata.h5
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from analysis.figures._style import COLORS, save_fig, use_style

DEFAULT_H5 = (
    "/data2/scratch/group_scratch/idr_plm/0000_dump/rsync_sherlock/AFDB/AFDB/"
    "AFDB_v4_idr_alldata/clustering/AFDB_IDR_90_alldata.h5"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--name", default="corpus_length_distributions")
    args = ap.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    with h5py.File(args.h5, "r") as f:
        idr = f["idr_length"][:].astype(np.int32)
        full = f["full_length"][:].astype(np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.hist(idr, bins=120, range=(0, 600), color=COLORS["darkblue"])
    ax1.set_xlabel("IDR length (residues)")
    ax1.set_ylabel("count")
    ax1.set_title(f"IDRs (n = {len(idr):,})", fontsize=18)

    ax2.hist(full, bins=120, range=(0, 1500), color=COLORS["lightblue"])
    for cut in (512, 1024):
        ax2.axvline(cut, color=COLORS["red"], lw=1.5, ls="--")
        ax2.text(cut + 10, ax2.get_ylim()[1] * 0.92, str(cut), color=COLORS["red"], fontsize=14)
    kept = 100 * np.mean(full <= 1020)
    ax2.set_xlabel("protein length (residues)")
    ax2.set_ylabel("count")
    ax2.set_title(f"proteins (≤1020: {kept:.0f}%)", fontsize=18)

    fig.tight_layout()
    out = save_fig(fig, args.name, subdir="si_figs/dataset")
    print("saved", out)


if __name__ == "__main__":
    main()
