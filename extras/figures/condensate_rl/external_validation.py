"""Lab-journal figure: orthogonal external validation of the GRPO runs.

Two independent predictors (neither trained on ProtGPS labels), one panel each:
  Left  — DeepLoc organelle axis: mean P(Nucleus) and P(Cytoplasm) per model. The nuclear targets
          (chromosome, nucleolus) shift nuclear; the cytoplasmic granules (p-body, stress_granule) do
          not read cytoplasmic (DeepLoc has no condensate class).
  Right — catGRANULE LLPS axis: mean phase-separation propensity per model. This rescues exactly the
          cytoplasmic granules DeepLoc misses — they are archetypal membraneless condensates.
Base (pre-RL) is shown as the reference baseline in both.

  IDIOM_FIG_DIR=.../lab_journal IDIOM_CONDENSATE_RESULTS=.../04_condensate \
      python -m extras.figures.condensate_rl.external_validation
"""

from __future__ import annotations

import argparse

import numpy as np

from extras.figures._style import COLORS, row_fig, save_fig, use_style
from extras.figures.condensate_rl._common import MODELS, load_deeploc, load_llps, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--name", default="external_validation")
    args = ap.parse_args()

    use_style()
    res = results_dir(args.results)
    dl_comps, dl = load_deeploc(res)
    llps_mean, llps_frac = load_llps(res)
    nuc = dl_comps.index("Nucleus")
    cyt = dl_comps.index("Cytoplasm")

    fig, (axL, axR) = row_fig(2)
    x = np.arange(len(MODELS))
    w = 0.38

    axL.bar(x - w / 2, [dl[m][nuc] for m in MODELS], w, label="P(Nucleus)", color=COLORS["darkblue"])
    axL.bar(x + w / 2, [dl[m][cyt] for m in MODELS], w, label="P(Cytoplasm)", color=COLORS["orange"])
    axL.set_xticks(x); axL.set_xticklabels(MODELS, rotation=20, ha="right")
    axL.set_ylabel("mean DeepLoc probability"); axL.set_ylim(0, 1)
    axL.set_title("DeepLoc: nuclear targets validate")
    axL.legend(frameon=False)

    bars = axR.bar(x, [llps_mean[m] for m in MODELS], 0.6, color=COLORS["green"])
    axR.axhline(0.5, ls="--", lw=1, color=COLORS["darkgrey"], label="LLPS threshold")
    for xi, m in zip(x, MODELS):
        axR.text(xi, llps_mean[m] + 0.02, f"{100*llps_frac[m]:.0f}%", ha="center", fontsize=8)
    axR.set_xticks(x); axR.set_xticklabels(MODELS, rotation=20, ha="right")
    axR.set_ylabel("mean catGRANULE LLPS score"); axR.set_ylim(0, 1)
    axR.set_title("catGRANULE: condensate targets validate")
    axR.legend(frameon=False)

    path = save_fig(fig, args.name, subdir="journal_figs")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
