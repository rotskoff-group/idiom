"""SI figure: IDR and full-protein length distributions of the final training corpus, vs DisProt.

Lengths are read from the final curation FASTA (one record per IDR): IDR length from the `_IDR_x-y`
header span, full-protein length from the sequence. Overlaid is DisProt under the *same* filter used
in `composition_vs_disprot.py` (`disprot_regions`: non-IDP, IDR >= 30, full seq <= 1020) — one entry
per qualifying 'D' region, matching the corpus' per-IDR-record convention. Distributions are
density-normalized because DisProt is ~10^3x smaller than the corpus. The 512 / 1024 context budgets
are marked on the protein-length panel (the corpus is already capped at <= 1024 - 4 = 1020).

    python analysis/figures/pretraining_data/corpus_lengths.py --n 2000000
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from extras.figures._style import COLORS, row_fig, save_fig, use_style
from extras.figures.pretraining_data.composition_vs_disprot import disprot_regions
from extras.figures.pretraining_data.corpus_composition import iter_records
from extras.figures.pretraining_data.final_corpus import FINAL_FASTA

DATA = os.environ.get("IDIOM_DATA", "/path/to/idiom_data")
DEFAULT_DISPROT = f"{DATA}/reference/disprot/DisProt release_2025_06 with_ambiguous_evidences.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=FINAL_FASTA)
    ap.add_argument("--disprot-json", default=DEFAULT_DISPROT)
    ap.add_argument("--n", type=int, default=2_000_000, help="records to sample (contiguous)")
    ap.add_argument("--name", default="corpus_length_distributions")
    args = ap.parse_args()

    use_style()

    idr, full = [], []
    for _acc, seq, s, e in itertools.islice(iter_records(args.fasta), args.n):
        idr.append(e - s)
        full.append(len(seq))
    idr = np.array(idr, dtype=np.int32)
    full = np.array(full, dtype=np.int32)

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
    ax2.set_xlabel("protein length (residues)")
    ax2.set_ylabel("density")
    ax2.set_title("proteins (all ≤1020 by construction)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    out = save_fig(fig, args.name, subdir="si_figs/dataset")
    print("saved", out, f"| DisProt regions: {len(dp):,}")


if __name__ == "__main__":
    main()
