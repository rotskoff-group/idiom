"""SI figure: pLDDT distributions of the extracted AFDB corpus (sampled).

Left: mean pLDDT over each IDR (the disordered regions; window-15 smoothed). Right: the max
per-residue pLDDT over each full protein — proteins whose max is below the folded threshold (80)
have no confident/folded region, i.e. the 'fully low-pLDDT' set that is dropped in curation.

    python analysis/figures/corpus_plddt.py --h5 /path/AFDB_IDR_90_alldata.h5 --n 200000
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from analysis.figures._style import COLORS, save_fig, use_style

DEFAULT_H5 = (
    "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/pretraining/AFDB/"
    "clustering_90/AFDB_IDR_90_alldata.h5"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--n", type=int, default=200_000, help="records to sample (contiguous)")
    ap.add_argument("--name", default="corpus_plddt")
    args = ap.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    with h5py.File(args.h5, "r") as f:
        idr = f["idr_plddt"][: args.n]
        full = f["full_avg_plddt"][: args.n]
    mean_idr = np.array([x.mean() for x in idr], dtype=np.float32)
    max_full = np.array([x.max() for x in full], dtype=np.float32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.hist(mean_idr, bins=80, range=(0, 100), color=COLORS["darkblue"])
    ax1.axvline(70, color=COLORS["red"], lw=1.5, ls="--")
    ax1.set_xlabel("mean IDR pLDDT")
    ax1.set_ylabel("count")
    ax1.set_title(f"IDRs (n = {len(mean_idr):,})", fontsize=18)

    ax2.hist(max_full, bins=80, range=(0, 100), color=COLORS["lightblue"])
    ax2.axvline(80, color=COLORS["red"], lw=1.5, ls="--")
    frac = 100 * np.mean(max_full < 80)
    ax2.set_xlabel("max protein pLDDT")
    ax2.set_ylabel("count")
    ax2.set_title(f"proteins (max < 80: {frac:.0f}%)", fontsize=18)

    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
