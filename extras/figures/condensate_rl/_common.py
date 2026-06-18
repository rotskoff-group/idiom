"""Shared loaders for the condensate-RL (ProtGPS-GRPO validation) figures.

All figures read the canonical results layout written by ``extras.eval.run_condensate`` (or the
converted equivalent): ``<results>/scores/{protgps,deeploc,llps}/<model>.csv`` +
``<results>/scores/naturalness.csv``. The results dir defaults to ``$IDIOM_CONDENSATE_RESULTS``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

# Display order; "base" is the pre-RL control. The 4 GRPO targets follow.
MODELS = ["base", "chromosome", "nucleolus", "p-body", "stress_granule"]
# ProtGPS condensate target -> expected DeepLoc organelle class (None = control).
EXPECTED_ORGANELLE = {"base": None, "chromosome": "Nucleus", "nucleolus": "Nucleus",
                      "p-body": "Cytoplasm", "stress_granule": "Cytoplasm"}


def results_dir(arg: str | None = None) -> Path:
    d = arg or os.environ.get("IDIOM_CONDENSATE_RESULTS")
    if not d:
        raise RuntimeError("Set --results or $IDIOM_CONDENSATE_RESULTS to the condensate-RL results dir.")
    return Path(d)


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = np.array([[float(x) for x in row] for row in r], dtype=float)
    return header, rows


def load_protgps(results: Path, models=MODELS):
    """Returns (compartments[list], means[model -> np.ndarray over compartments])."""
    comps, means = None, {}
    for m in models:
        header, rows = _read_csv(results / "scores" / "protgps" / f"{m}.csv")
        comps = header[1:]  # drop 'idx'
        means[m] = rows[:, 1:].mean(axis=0)
    return comps, means


def load_deeploc(results: Path, models=MODELS):
    """Returns (compartments[list], means[model -> np.ndarray]) of DeepLoc per-class mean prob."""
    comps, means = None, {}
    for m in models:
        header, rows = _read_csv(results / "scores" / "deeploc" / f"{m}.csv")
        comps = header[1:]
        means[m] = np.nanmean(rows[:, 1:], axis=0)
    return comps, means


def load_llps(results: Path, models=MODELS, threshold: float = 0.5):
    """Returns mean[model] and frac_ge[model] of catGRANULE LLPS score."""
    mean, frac = {}, {}
    for m in models:
        _, rows = _read_csv(results / "scores" / "llps" / f"{m}.csv")
        s = rows[:, 1]
        s = s[~np.isnan(s)]
        mean[m] = float(s.mean())
        frac[m] = float((s >= threshold).mean())
    return mean, frac


def load_naturalness(results: Path):
    """Returns (columns[list], rows[dict]) from scores/naturalness.csv (one row per set)."""
    with open(results / "scores" / "naturalness.csv") as fh:
        rows = list(csv.DictReader(fh))
    return list(rows[0]), rows
