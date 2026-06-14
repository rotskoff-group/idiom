"""pLDDT-based IDR segmentation (the pure core of curation stage 1).

Ported from the legacy ``extract_idr_alldata.py``, returning **0-based half-open** spans
(``idr = full_seq[start:end]``, D16) so the result feeds :class:`idiom.data.io.Record`
directly. Pure and CPU-testable; the AFDB-h5 driver that calls this over real data is in the
operator runbook (heavy, not run during training/CI).

Algorithm (per protein): window-average the per-residue pLDDT, label each residue
folded / disordered / gap by threshold, drop too-short segments, relabel gaps by their
neighbours, merge adjacent same-label runs, then keep disordered runs within the length band.
"""

from __future__ import annotations

import numpy as np


def _windowed_mean(plddt: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(plddt, pad, mode="edge")  # edge-pad so the output keeps the input length
    return np.convolve(padded, np.ones(window) / window, mode="valid")  # centered moving average


def extract_idrs(
    plddt: np.ndarray,
    *,
    folded_thresh: float = 80.0,
    disordered_thresh: float = 70.0,
    min_seg_length: int = 10,
    min_idr_length: int = 30,
    max_idr_length: int = 4096,
    window: int = 15,
) -> list[tuple[int, int]]:
    """Return IDR spans as 0-based half-open ``(start, end)`` pairs (``idr = seq[start:end]``)."""
    plddt = np.asarray(plddt, dtype=float)
    avg = _windowed_mean(plddt, window)
    n = len(avg)
    if n == 0:
        return []

    labels = np.empty(n, dtype=object)
    labels[avg > folded_thresh] = "folded"
    labels[avg < disordered_thresh] = "disordered"
    labels[(avg >= disordered_thresh) & (avg <= folded_thresh)] = "gap"

    # contiguous (label, start, end_inclusive) runs
    segments: list[tuple[str, int, int]] = []
    cur, s0 = labels[0], 0
    for i in range(1, n):
        if labels[i] != cur:
            segments.append((cur, s0, i - 1))
            cur, s0 = labels[i], i
    segments.append((cur, s0, n - 1))

    # too-short folded/disordered runs become gaps
    segs = [
        ("gap", s, e) if lab in ("folded", "disordered") and (e - s + 1) < min_seg_length else (lab, s, e)
        for lab, s, e in segments
    ]

    # relabel gaps by their neighbours (disordered only if a disordered neighbour touches them)
    relabeled: list[tuple[str, int, int]] = []
    for idx, (lab, s, e) in enumerate(segs):
        if lab != "gap":
            relabeled.append((lab, s, e))
            continue
        left = segs[idx - 1][0] if idx > 0 else None
        right = segs[idx + 1][0] if idx < len(segs) - 1 else None
        if len(segs) == 1:
            new = "folded"
        elif idx == 0:
            new = "disordered" if right == "disordered" else "folded"
        elif idx == len(segs) - 1:
            new = "disordered" if left == "disordered" else "folded"
        else:
            new = "disordered" if left == "disordered" and right == "disordered" else "folded"
        relabeled.append((new, s, e))

    # merge adjacent same-label runs
    merged: list[tuple[str, int, int]] = []
    for lab, s, e in relabeled:
        if merged and merged[-1][0] == lab and s == merged[-1][2] + 1:
            plab, ps, _ = merged[-1]
            merged[-1] = (plab, ps, e)
        else:
            merged.append((lab, s, e))

    # keep disordered runs in the length band; return half-open spans
    out: list[tuple[int, int]] = []
    for lab, s, e in merged:
        if lab == "disordered" and min_idr_length <= (e - s + 1) <= max_idr_length:
            out.append((s, e + 1))  # inclusive end -> half-open
    return out
