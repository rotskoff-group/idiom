"""Per-sequence biophysical features via `sparrow` — the IDR eval feature set.

Single source of truth ported from the v1 `idr-plm-figures` ``utils.calculate_sparrow_metrics``;
shared by `python -m eval.run` and the manuscript figures so they compute identically.
"""

from __future__ import annotations

import numpy as np

# Canonical feature set (charge, composition, complexity, patterning).
FEATURES: tuple[str, ...] = (
    "length", "FCR", "NCPR", "fraction_positive", "fraction_negative",
    "fraction_aromatic", "fraction_aliphatic", "fraction_polar", "fraction_proline",
    "complexity", "kappa", "SCD", "SHD",
)


def sequence_features(seq: str) -> dict[str, float]:
    """All FEATURES for one sequence (NaN for each on failure). ``kappa`` is -1 when undefined
    (sequence lacks both + and - residues) — that's sparrow's sentinel, kept as-is."""
    from sparrow import Protein

    try:
        p = Protein(seq)
        return {
            "length": float(len(p.sequence)),
            "FCR": float(p.FCR), "NCPR": float(p.NCPR),
            "fraction_positive": float(p.fraction_positive), "fraction_negative": float(p.fraction_negative),
            "fraction_aromatic": float(p.fraction_aromatic), "fraction_aliphatic": float(p.fraction_aliphatic),
            "fraction_polar": float(p.fraction_polar), "fraction_proline": float(p.fraction_proline),
            "complexity": float(p.complexity),
            "kappa": float(p.kappa), "SCD": float(p.SCD), "SHD": float(p.SHD),
        }
    except Exception:
        return {k: float("nan") for k in FEATURES}


def features_table(seqs: list[str], features: tuple[str, ...] = FEATURES) -> dict[str, np.ndarray]:
    """``list[str]`` -> ``{feature: np.ndarray}`` (one row per sequence, NaNs kept)."""
    rows = [sequence_features(s) for s in seqs]
    return {f: np.array([r[f] for r in rows], dtype=float) for f in features}


def summarize(table: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per-feature mean/std/median over a features table (NaNs ignored)."""
    out: dict[str, dict[str, float]] = {}
    for f, arr in table.items():
        a = arr[~np.isnan(arr)]
        out[f] = {
            "mean": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std()) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
            "n": int(a.size),
        }
    return out


# --- composition extras (naturalness / reward-hacking checks; kept separate from sparrow FEATURES) ---
# Residues characteristic of condensate / low-complexity IDRs (R/G/Q/N/S/Y).
LCD_RESIDUES = "RGQNSY"
COMPOSITION_EXTRAS: tuple[str, ...] = ("aa_entropy", "lcd_fraction")


def aa_entropy(seq: str) -> float:
    """Shannon entropy (bits) of the amino-acid composition — low values flag low-complexity tracts."""
    if not seq:
        return float("nan")
    import math  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in Counter(seq).values())


def lcd_fraction(seq: str, residues: str = LCD_RESIDUES) -> float:
    """Fraction of residues in the low-complexity/condensate set (default R/G/Q/N/S/Y)."""
    if not seq:
        return float("nan")
    return sum(seq.count(r) for r in residues) / len(seq)


def composition_extras_table(seqs: list[str]) -> dict[str, np.ndarray]:
    """``{aa_entropy, lcd_fraction}`` arrays (one row per sequence) — pairs with ``features_table``."""
    return {
        "aa_entropy": np.array([aa_entropy(s) for s in seqs], dtype=float),
        "lcd_fraction": np.array([lcd_fraction(s) for s in seqs], dtype=float),
    }
