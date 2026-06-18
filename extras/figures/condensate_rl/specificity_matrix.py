"""Lab-journal figure: ProtGPS target-selectivity ("specificity") matrix for the GRPO runs.

Heatmap of mean ProtGPS P(compartment) for each model (rows) across all 12 condensate compartments
(cols). Diagonal dominance — each GRPO model peaking on *its own* target while off-targets stay low
— is the evidence that the reward is target-selective rather than a generic condensate push. ProtGPS
is in-distribution (it *is* the reward), so this is a selectivity check, not an independent validator.

  IDIOM_FIG_DIR=.../lab_journal IDIOM_CONDENSATE_RESULTS=.../04_condensate \
      python -m extras.figures.condensate_rl.specificity_matrix
"""

from __future__ import annotations

import argparse

import numpy as np

from extras.figures._style import save_fig, use_style
from extras.figures.condensate_rl._common import MODELS, load_protgps, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--name", default="protgps_specificity")
    args = ap.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    comps, means = load_protgps(results_dir(args.results))
    mat = np.stack([means[m] for m in MODELS], axis=0)  # [n_models, n_comps]

    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("ProtGPS compartment (scored)")
    ax.set_ylabel("GRPO model (reward target)")
    # annotate; box the on-target (own-name) cell of each GRPO model
    for i, m in enumerate(MODELS):
        for j, c in enumerate(comps):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.55 else "black")
            if c == m:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#01abe9", lw=2.2))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="mean P(compartment)")
    ax.set_title("ProtGPS specificity: each GRPO model is selectively elevated on its own target")
    path = save_fig(fig, args.name, subdir="journal_figs")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
