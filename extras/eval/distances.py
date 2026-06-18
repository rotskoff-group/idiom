"""Distributional distances between a generated set and a reference (e.g. DisProt).

Wasserstein-1 per feature — the paper's headline generated-vs-natural comparison (Fig 2). Computed
directly on the samples (no histogram binning), plus a reference-normalized variant so features on
different scales are comparable.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def w1(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 between two sample arrays (NaNs dropped)."""
    a, b = _clean(a), _clean(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def w1_normalized(gen: np.ndarray, ref: np.ndarray) -> float:
    """``w1(gen, ref)`` divided by the reference's std, so per-feature distances are comparable."""
    ref_std = _clean(ref).std()
    d = w1(gen, ref)
    return d / ref_std if ref_std > 0 else d


def w1_table(
    gen: dict[str, np.ndarray], ref: dict[str, np.ndarray], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Features tables (gen vs ref) -> ``{feature: {w1, w1_norm}}`` for shared features."""
    feats = features or [f for f in gen if f in ref]
    return {f: {"w1": w1(gen[f], ref[f]), "w1_norm": w1_normalized(gen[f], ref[f])} for f in feats}
