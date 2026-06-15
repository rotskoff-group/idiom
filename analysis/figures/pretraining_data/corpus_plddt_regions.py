"""SI figure: pLDDT of IDR vs non-IDR regions in the extracted AFDB corpus (sampled).

Complements `corpus_plddt.py`. For each sampled protein we split its `full_avg_plddt` (window-15
smoothed) into IDR residues (the union of *all* its extracted IDR spans) and non-IDR residues
(everything else — predominantly folded/flanking context). Records are grouped by protein so a
multi-IDR protein's other IDRs are not miscounted as "non-IDR".

  python analysis/figures/corpus_plddt_regions.py --h5 /path/AFDB_IDR_90_alldata.h5 --n 200000

Panels: (left) per-protein mean pLDDT, IDR vs non-IDR; (mid) pooled per-residue pLDDT, IDR vs
non-IDR; (right) joint per-protein mean non-IDR vs mean IDR pLDDT.
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
    ap.add_argument("--n", type=int, default=200_000, help="records to sample (contiguous)")
    ap.add_argument("--name", default="corpus_plddt_regions")
    args = ap.parse_args()

    use_style()

    with h5py.File(args.h5, "r") as f:
        acc = f["accession_ids"][: args.n]
        st = f["idr_start"][: args.n]
        en = f["idr_end"][: args.n]
        fap = f["full_avg_plddt"][: args.n]

    bases = [a.decode().rsplit("_", 1)[0] for a in acc]  # protein id, dropping the _start-end tag
    idr_mean, non_mean = [], []          # per-protein region means
    idr_res, non_res = [], []            # pooled per-residue values
    i, n = 0, len(bases)
    while i < n:
        j = i
        while j < n and bases[j] == bases[i]:  # rows of one protein are contiguous
            j += 1
        prot = np.asarray(fap[i], dtype=np.float32)  # whole-protein pLDDT (same across its rows)
        mask = np.zeros(len(prot), dtype=bool)
        for k in range(i, j):
            mask[int(st[k]) : int(en[k]) + 1] = True  # idr_end is inclusive
        idr_p, non_p = prot[mask], prot[~mask]
        if idr_p.size:
            idr_mean.append(idr_p.mean())
            idr_res.append(idr_p)
        if non_p.size:
            non_mean.append(non_p.mean())
            non_res.append(non_p)
        i = j

    idr_mean = np.array(idr_mean)
    non_mean = np.array(non_mean)
    idr_res = np.concatenate(idr_res)
    non_res = np.concatenate(non_res)

    fig, (ax1, ax2, ax3) = row_fig(3)
    blue, orange = COLORS["darkblue"], COLORS["orange"]

    # (a) per-protein mean pLDDT
    bins = np.linspace(0, 100, 81)
    ax1.hist(idr_mean, bins=bins, color=blue, alpha=0.65, label="IDR")
    ax1.hist(non_mean, bins=bins, color=orange, alpha=0.65, label="non-IDR")
    ax1.set_xlabel("mean pLDDT per protein region")
    ax1.set_ylabel("count")
    ax1.set_title(f"per-protein means (n = {len(idr_mean):,})")
    ax1.legend(frameon=False)

    # (b) pooled per-residue pLDDT (density: very different residue counts)
    ax2.hist(idr_res, bins=bins, density=True, color=blue, alpha=0.65, label="IDR")
    ax2.hist(non_res, bins=bins, density=True, color=orange, alpha=0.65, label="non-IDR")
    ax2.axvline(70, color=COLORS["grey"], lw=1.2, ls="--")
    ax2.axvline(80, color=COLORS["grey"], lw=1.2, ls="--")
    ax2.set_xlabel("per-residue pLDDT")
    ax2.set_ylabel("density")
    ax2.set_title(f"per-residue (n = {len(idr_res) + len(non_res):,})")
    ax2.legend(frameon=False)

    # (c) joint per-protein non-IDR vs IDR mean pLDDT
    m = min(len(idr_mean), len(non_mean))  # proteins with both an IDR and a non-IDR region
    h = ax3.hist2d(non_mean[:m], idr_mean[:m], bins=80, range=[[0, 100], [0, 100]], cmin=1)
    ax3.plot([0, 100], [0, 100], color=COLORS["grey"], lw=1.0, ls="--")
    ax3.set_xlabel("mean non-IDR pLDDT")
    ax3.set_ylabel("mean IDR pLDDT")
    ax3.set_title("joint (per protein)")
    fig.colorbar(h[3], ax=ax3, label="count")

    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
