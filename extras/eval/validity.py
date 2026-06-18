"""Generation-health metrics: did generations terminate, at what length, with canonical residues.

Operates on the decoded IDR residue strings that `generate()` returns (STOP already stripped). A
sequence that hit the `max_new_tokens` cap without emitting STOP has length == max_new_tokens; one
that terminated is shorter. (Heuristic — exact only when generations come from a single cap.)
"""

from __future__ import annotations

import numpy as np

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def validity_stats(seqs: list[str], max_new_tokens: int) -> dict[str, float]:
    if not seqs:
        return {"n": 0}
    lens = np.array([len(s) for s in seqs])
    terminated = lens < max_new_tokens
    noncanon = sum(sum(c not in CANONICAL for c in s) for s in seqs)
    return {
        "n": len(seqs),
        "frac_terminated": float(terminated.mean()),          # emitted <eos> before the cap
        "length_mean": float(lens.mean()),
        "length_median": float(np.median(lens)),
        "length_min": int(lens.min()),
        "length_max": int(lens.max()),
        "frac_noncanonical_residues": float(noncanon / max(int(lens.sum()), 1)),
        "frac_unique": float(len(set(seqs)) / len(seqs)),     # 1.0 = no exact-duplicate generations
    }
