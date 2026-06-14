"""Shared matplotlib style + figure saving for manuscript figures.

Figures are saved into the overleaf repo's ``figs/`` tree via the ``IDIOM_FIG_DIR`` env var,
named to match the LaTeX ``\\includegraphics`` path — so regenerating overwrites in place and
``main.tex`` picks it up with no edits. PDF (vector) by default for the .tex; PNG where the
manuscript uses raster (e.g. some SAE logos).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (srun / no display)
import matplotlib.pyplot as plt  # noqa: E402

STYLE = Path(__file__).with_name("idiom.mplstyle")

# Named palette (from the style's diverging/sequential sets) for ad-hoc use in figures.
COLORS = {
    "red": "#f54b1a", "pink": "#d19eb1", "orange": "#d3772e", "yellow": "#ebe85d",
    "green": "#0f6852", "lightblue": "#01abe9", "darkblue": "#1b346c", "tan": "#e5c39e",
    "darktan": "#af9e73", "grey": "#c3ced0", "darkgrey": "#9dadc4", "black": "#110d1b",
    "white": "#f1f8f1",
}
DIVERGING = ["#00386b", "#00679f", "#1b9ad0", "#54cffb", "#fbac41", "#fa8f34", "#f87026", "#f54b1a"]
SEQUENTIAL = ["#001f4d", "#00386b", "#00538a", "#0070a9", "#008eca", "#05adea", "#4dcbff", "#78ecff"]


def use_style() -> None:
    """Apply the IDiom manuscript matplotlib style. Call once at the top of a figure script."""
    plt.style.use(str(STYLE))


def fig_dir() -> Path:
    """The manuscript ``figs/`` directory (``IDIOM_FIG_DIR``)."""
    d = os.environ.get("IDIOM_FIG_DIR")
    if not d:
        raise RuntimeError(
            "Set IDIOM_FIG_DIR to the manuscript figs/ dir, e.g. "
            "/data2/scratch/jxliu2/papers/overleaf/IDiom-manuscript-v1/figs"
        )
    return Path(d)


def save_fig(fig, name: str, subdir: str = "si_figs", ext: str = "pdf") -> Path:
    """Save ``fig`` to ``$IDIOM_FIG_DIR/<subdir>/<name>.<ext>`` (the LaTeX path). Returns it."""
    out = fig_dir() / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.{ext}"
    fig.savefig(path, bbox_inches="tight")
    return path
