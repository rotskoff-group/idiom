"""Disorder prediction. Primary: metapredict (core dep, GPU-capable). Optional: IUPred3 (licensed
external download, path-referenced — skipped if unavailable).

Mean per-residue disorder and fraction-of-residues-disordered per sequence; for generations these
should track the natural-IDR references (the paper's disorder check, Fig 1g / SI).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from eval._tools import tool_path


def iupred3_disorder(
    seqs: list[str], *, iupred_path: str | None = None, mode: str = "long", threshold: float = 0.5
) -> dict[str, np.ndarray]:
    """Per-sequence IUPred3 ``mean_disorder`` + ``frac_disordered`` — orthogonal to metapredict.

    Optional external tool (academic-license download, not a pip dep): ``iupred_path`` is the
    extracted ``iupred3/`` dir (importable, ships its own ``data/``); falls back to ``$IUPRED3_PATH``
    then the known location. Raises ImportError if unavailable.
    """
    path = iupred_path or os.environ.get("IUPRED3_PATH") or tool_path("iupred3")
    if path not in sys.path:
        sys.path.insert(0, path)
    import iupred3_lib  # noqa: PLC0415

    means, fracs = [], []
    for s in seqs:
        try:  # IUPred's savgol smoothing fails on sequences shorter than its window
            scores = np.asarray(iupred3_lib.iupred(s, mode)[0], dtype=float)
            means.append(float(scores.mean()) if scores.size else np.nan)
            fracs.append(float((scores > threshold).mean()) if scores.size else np.nan)
        except Exception:
            means.append(np.nan)
            fracs.append(np.nan)
    return {"mean_disorder": np.array(means), "frac_disordered": np.array(fracs)}


def metapredict_disorder(
    seqs: list[str], *, threshold: float = 0.5, device: str | None = None, version: str = "V3"
) -> dict[str, np.ndarray]:
    """Per-sequence ``mean_disorder`` and ``frac_disordered`` (residues with score > threshold)."""
    import metapredict as meta

    preds = meta.predict_disorder_batch(seqs, device=device, version=version)  # [[seq, scores], ...]
    arrs = [np.asarray(p[1], dtype=float) for p in preds]
    means = np.array([a.mean() if a.size else np.nan for a in arrs])
    fracs = np.array([(a > threshold).mean() if a.size else np.nan for a in arrs])
    return {"mean_disorder": means, "frac_disordered": fracs}


def disorder_stats(scores: dict[str, np.ndarray]) -> dict[str, float]:
    md = scores["mean_disorder"][~np.isnan(scores["mean_disorder"])]
    fd = scores["frac_disordered"][~np.isnan(scores["frac_disordered"])]
    return {
        "mean_disorder": float(md.mean()) if md.size else float("nan"),
        "median_disorder": float(np.median(md)) if md.size else float("nan"),
        "frac_disordered_mean": float(fd.mean()) if fd.size else float("nan"),
        "n": int(md.size),
    }
