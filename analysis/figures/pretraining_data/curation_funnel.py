"""SI figure: record-count funnel across the v2 curation pipeline.

A single horizontal funnel of the measured record counts at every curation stage, from the raw
AFDB IDR extraction down to the final training corpus. All counts are measured from the files (see
`data_pipeline/RUNBOOK.md` and the lab journal); they are recorded here as provenance constants so
the figure is self-contained and reproduces the exact pipeline numbers without re-reading the 25 GB
FASTAs. Bars are annotated with the absolute count and the fraction retained relative to the raw
extraction; the per-stage drop is shown as a faded "removed" segment.

  python analysis/figures/pretraining_data/curation_funnel.py
"""

from __future__ import annotations

import argparse

import numpy as np

from analysis.figures._style import COLORS, row_fig, save_fig, use_style

# (label, records) — measured provenance, raw extraction first.
STAGES = [
    ("AFDB v4 raw IDRs", 110_464_867),          # 64 extraction parts (Tesei pLDDT segmentation)
    ("90% cluster representatives", 73_043_918),  # mmseqs linclust 90% / 80% cov; curated master h5
    ("length ≤1020 + folded segment", 57_788_195),  # stage 3: drop len>1020 and fully-low-pLDDT
    ("DisProt dedup", 57_732_276),               # stage 4: drop ≥50% id to a benchmark DisProt IDR
    ("signal-peptide filter", 54_155_136),       # stage 4b: SignalP-6 trim/drop (final training corpus)
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="curation_funnel")
    args = ap.parse_args()

    use_style()

    labels = [s[0] for s in STAGES]
    counts = np.array([s[1] for s in STAGES], dtype=float)
    raw = counts[0]
    millions = counts / 1e6
    y = np.arange(len(STAGES))[::-1]  # first stage on top

    fig, ax = row_fig(1)
    # full bar (faded) = previous-stage size; solid overlay = records that survive this stage.
    prev = np.concatenate([[raw], counts[:-1]])
    ax.barh(y, prev / 1e6, color=COLORS["grey"], height=0.62, zorder=1)
    ax.barh(y, millions, color=COLORS["darkblue"], height=0.62, zorder=2)

    for yi, (m, c, p) in zip(y, zip(millions, counts, prev)):
        drop = f"  ·  −{100 * (p - c) / p:.1f}% this stage" if p > c else ""
        ax.text(p / 1e6 + raw / 1e6 * 0.015, yi,  # right of the faded previous-stage bar
                f"{c:,.0f}  ({100 * c / raw:.0f}% of raw){drop}",
                va="center", ha="left", color=COLORS["black"], fontsize=14)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("records (millions)")
    ax.set_xlim(0, raw / 1e6 * 1.95)
    ax.margins(y=0.04)
    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
