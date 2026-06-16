"""Novelty / memorization: max %identity of each generation to the training corpus (mmseqs).

Mirrors the v1 analysis (``mmseqs search`` of generations against the training IDRs, closest hit
per query). Wraps the **mmseqs binary** (not a pip dep): pass ``--mmseqs`` / ``$MMSEQS`` or it falls
back to PATH then the known static-binary location. A generation with a near-1.0 max identity is a
memorized training sequence.
"""

from __future__ import annotations

import shutil
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from eval._tools import tool_path


def _resolve_mmseqs(mmseqs: str | None) -> str:
    return mmseqs or os.environ.get("MMSEQS") or shutil.which("mmseqs") or tool_path("bin", "mmseqs")


def max_identity(
    query_seqs: list[str], target_fasta: str, *, mmseqs: str | None = None,
    sensitivity: float = 7.0, threads: int = 8, tmp_dir: str | None = None,
) -> np.ndarray:
    """For each query, fraction identity to its closest match in `target_fasta` (0.0 if no hit).

    Returns an array aligned to `query_seqs` order. Uses ``mmseqs easy-search`` with ``--max-seqs 1``
    (best hit per query), permissive thresholds so even distant matches register.
    """
    binary = _resolve_mmseqs(mmseqs)
    work = Path(tempfile.mkdtemp(dir=tmp_dir, prefix="idiom_novelty_"))
    try:
        qfa = work / "query.fasta"
        with qfa.open("w") as f:
            for i, s in enumerate(query_seqs):
                f.write(f">q{i}\n{s}\n")
        res = work / "hits.tsv"
        subprocess.run(
            [binary, "easy-search", str(qfa), target_fasta, str(res), str(work / "tmp"),
             "--max-seqs", "1", "-s", str(sensitivity), "--min-seq-id", "0", "-c", "0",
             "--threads", str(threads), "--format-output", "query,fident", "-v", "1"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        best: dict[int, float] = {}
        with res.open() as f:
            for line in f:
                q, fident = line.split("\t")
                qi = int(q[1:])
                best[qi] = max(best.get(qi, 0.0), float(fident))
        return np.array([best.get(i, 0.0) for i in range(len(query_seqs))], dtype=float)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def novelty_stats(identities: np.ndarray, thresholds=(0.5, 0.9, 0.95)) -> dict[str, float]:
    """Summarize the max-identity distribution; ``frac_ge_T`` = fraction of generations within T of a
    training sequence (high ``frac_ge_0.9`` ⇒ memorization)."""
    out = {
        "max_identity_mean": float(identities.mean()) if identities.size else float("nan"),
        "max_identity_median": float(np.median(identities)) if identities.size else float("nan"),
    }
    for t in thresholds:
        out[f"frac_ge_{t}"] = float((identities >= t).mean()) if identities.size else float("nan")
    return out
