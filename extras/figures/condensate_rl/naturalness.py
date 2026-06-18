"""Lab-journal figure: naturalness / reward-hacking control for the GRPO runs.

Small multiples comparing each model (incl. the base control) against a natural-IDR reference on
metrics that would expose degenerate, reward-hacked sequences: sequence complexity, AA Shannon
entropy, and de-novo perplexity under the base model should all sit at natural levels (no collapse),
while LCD fraction / aromatic content shift in compartment-appropriate, interpretable ways. The
natural reference (``natural_ref``) is drawn as a dashed line.

  IDIOM_FIG_DIR=.../lab_journal IDIOM_CONDENSATE_RESULTS=.../04_condensate \
      python -m extras.figures.condensate_rl.naturalness
"""

from __future__ import annotations

import argparse

from extras.figures._style import COLORS, row_fig, save_fig, use_style
from extras.figures.condensate_rl._common import MODELS, load_naturalness, results_dir

# (column, label, "natural levels expected?") — the first three are the no-reward-hacking checks.
PANELS = [
    ("complexity", "sequence complexity", True),
    ("aa_entropy", "AA entropy (bits)", True),
    ("base_ppl", "base-model perplexity", True),
    ("lcd_fraction", "LCD fraction (RGQNSY)", False),
    ("fraction_aromatic", "aromatic fraction", False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--name", default="naturalness")
    args = ap.parse_args()

    use_style()
    _, rows = load_naturalness(results_dir(args.results))
    by = {r["model"]: r for r in rows}
    natural = by.get("natural_ref") or by.get("natural_AFDB")

    fig, axes = row_fig(len(PANELS))
    for ax, (col, label, natural_check) in zip(axes, PANELS):
        vals = [float(by[m][col]) for m in MODELS]
        colors = [COLORS["grey"] if m == "base" else COLORS["darkblue"] for m in MODELS]
        ax.bar(range(len(MODELS)), vals, 0.7, color=colors)
        if natural is not None:
            nv = float(natural[col])
            ax.axhline(nv, ls="--", lw=1.2, color=COLORS["red"],
                       label="natural IDRs" if ax is axes[0] else None)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8)
        title = label + ("  ✓natural" if natural_check else "  (target-specific)")
        ax.set_title(title, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Naturalness: complexity / entropy / perplexity stay at natural-IDR levels "
                 "(no reward hacking); composition shifts are compartment-specific", fontsize=11)
    fig.subplots_adjust(wspace=0.42, top=0.82)
    path = save_fig(fig, args.name, subdir="journal_figs")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
