"""SI figure: fraction of each protein that is intrinsically disordered (extracted AFDB corpus).

Per IDR record, the IDR fraction = `idr_length / full_length` (%). The analogue of the legacy
`pct_idr.py` (which measured the share of the FIM sequence after the `2` middle marker). The spike
at 100% is the whole-protein IDRs (fully disordered = the fully-low-pLDDT set removed in curation),
so the right panel excludes them to show the distribution over partially-disordered proteins.

  python analysis/figures/corpus_idr_fraction.py --h5 /path/AFDB_IDR_90_alldata.h5 --n 1000000
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from analysis.figures._style import COLORS, row_fig, save_fig, use_style

DEFAULT_H5 = (
    "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/pretraining/AFDB/"
    "clustering_90/AFDB_IDR_90_alldata.h5"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--n", type=int, default=1_000_000, help="records to sample (contiguous)")
    ap.add_argument("--name", default="corpus_idr_fraction")
    args = ap.parse_args()

    use_style()

    with h5py.File(args.h5, "r") as f:
        idr_len = f["idr_length"][: args.n].astype(np.float64)
        full_len = f["full_length"][: args.n].astype(np.float64)
    pct = 100.0 * idr_len / full_len  # IDR share of the protein, per record
    partial = pct[pct < 100.0]
    frac_full = 100.0 * np.mean(pct >= 100.0)

    fig, (ax1, ax2) = row_fig(2)

    ax1.hist(pct, bins=np.linspace(0, 100, 51), color=COLORS["darkblue"])
    ax1.set_xlabel("IDR fraction of protein (%)")
    ax1.set_ylabel("count")
    ax1.set_title(f"all records (n = {len(pct):,}; {frac_full:.0f}% are whole-protein)")

    ax2.hist(partial, bins=np.linspace(0, 100, 51), color=COLORS["lightblue"])
    ax2.set_xlabel("IDR fraction of protein (%)")
    ax2.set_ylabel("count")
    ax2.set_title(f"excluding whole-protein IDRs (n = {len(partial):,})")

    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
