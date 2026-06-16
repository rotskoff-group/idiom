"""SI figure: fraction of each protein that is intrinsically disordered (final training corpus).

Per IDR record, the IDR fraction = `idr_length / full_length` (%), read from the final curation
FASTA (IDR length from the `_IDR_x-y` header span, full length from the sequence). The analogue of
the legacy `pct_idr.py`. Unlike the raw extraction, the final corpus contains essentially no
whole-protein (100% disordered) records --- the fully-low-pLDDT filter removed proteins with no
folded segment --- so a single panel suffices; the distribution peaks at a small disordered fraction.

  python analysis/figures/pretraining_data/corpus_idr_fraction.py --n 1000000
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from analysis.figures._style import COLORS, row_fig, save_fig, use_style
from analysis.figures.pretraining_data.corpus_composition import iter_records
from analysis.figures.pretraining_data.final_corpus import FINAL_FASTA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=FINAL_FASTA)
    ap.add_argument("--n", type=int, default=1_000_000, help="records to sample (contiguous)")
    ap.add_argument("--name", default="corpus_idr_fraction")
    args = ap.parse_args()

    use_style()

    idr_len, full_len = [], []
    for _acc, seq, s, e in itertools.islice(iter_records(args.fasta), args.n):
        idr_len.append(e - s)
        full_len.append(len(seq))
    pct = 100.0 * np.array(idr_len, dtype=np.float64) / np.array(full_len, dtype=np.float64)
    frac_full = 100.0 * np.mean(pct >= 100.0)
    median = np.median(pct)

    fig, ax = row_fig(1)
    ax.hist(pct, bins=np.linspace(0, 100, 51), color=COLORS["darkblue"])
    ax.axvline(median, color=COLORS["red"], lw=1.5, ls="--")
    ax.text(median + 2, ax.get_ylim()[1] * 0.92, f"median {median:.0f}%", color=COLORS["red"], fontsize=14)
    ax.set_xlabel("IDR fraction of protein (%)")
    ax.set_ylabel("count")
    ax.set_title(f"n = {len(pct):,}  ({frac_full:.1f}% whole-protein)")

    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
