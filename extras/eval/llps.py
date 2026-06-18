"""Liquid-liquid phase-separation (LLPS) propensity via catGRANULE 2.0 ROBOT — the condensate axis
that complements organelle localization (:mod:`extras.eval.localization`).

catGRANULE 2.0 needs a pinned, old dependency stack (scikit-learn 1.1.1 to match its pickled
classifiers), so it lives in its own cloned repo + venv and runs as a subprocess: we hand it a FASTA
and an inline driver that calls ``compute_score_profile_fatsa_DF`` (physico-chemical / RandomForest
"FASTA mode", no structure needed) and writes ``name,LLPS_Score``. Score is a probability in [0, 1];
higher = more phase-separation prone (TDP-43 positive control ~0.87). The trained classifier uses a
0.5 decision threshold.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile

import numpy as np

from extras.eval._tools import catgranule_dir, catgranule_python

# Inline driver run *inside* the catGRANULE venv, cwd=repo (the tool relies on ./src relative paths).
_DRIVER = """
import sys
sys.path.insert(0, ".")
from compute_profiles_and_predictions import compute_score_profile_fatsa_DF
df = compute_score_profile_fatsa_DF(sys.argv[1])
df[["Name", "LLPS_Score"]].to_csv(sys.argv[2], index=False)
"""


def catgranule_score(seqs: list[str]) -> np.ndarray:
    """Per-sequence catGRANULE 2.0 LLPS score in [0, 1] (order preserved; empties dropped)."""
    kept = [s for s in seqs if s]
    if not kept:
        return np.empty(0)
    repo, py = catgranule_dir(), catgranule_python()
    with tempfile.TemporaryDirectory() as td:
        fasta, out_csv = os.path.join(td, "in.fasta"), os.path.join(td, "out.csv")
        with open(fasta, "w") as fh:
            for i, s in enumerate(kept):
                fh.write(f">seq_{i}\n{s}\n")
        subprocess.run([py, "-c", _DRIVER, fasta, out_csv], check=True, cwd=repo,
                       stdout=subprocess.DEVNULL)
        by_id = {r["Name"]: float(r["LLPS_Score"]) for r in csv.DictReader(open(out_csv))}
    return np.array([by_id.get(f"seq_{i}", np.nan) for i in range(len(kept))])


def llps_stats(scores: np.ndarray, *, threshold: float = 0.5) -> dict:
    s = scores[~np.isnan(scores)]
    return {
        "n": int(s.size),
        "mean_llps": float(s.mean()) if s.size else float("nan"),
        "median_llps": float(np.median(s)) if s.size else float("nan"),
        "frac_ge_threshold": float((s >= threshold).mean()) if s.size else float("nan"),
        "threshold": threshold,
    }
