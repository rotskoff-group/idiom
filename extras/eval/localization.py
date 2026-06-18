"""Subcellular localization via DeepLoc 2.1 — an orthogonal (ProtGPS-independent) check that
GRPO-steered IDPs land in the expected compartment.

DeepLoc ships its own venv (its torch/transformers differ from ours), so it runs as a subprocess
(``deeploc2`` CLI) on a temporary FASTA; we parse its ``results_*.csv`` back into per-sequence
compartment probabilities. The ``Fast`` (ESM1b) model is the default because its backbone is cached
in the tools ``torch_cache`` (no download); ``Accurate`` (ProtT5) is not supported here — the bundled
transformers refuses to load the ProtT5 ``.bin`` checkpoint (``torch.load`` CVE gate).

ProtGPS condensate target -> expected DeepLoc organelle class (the two relevant ones):
``chromosome``/``nucleolus`` -> ``Nucleus`` (nuclear condensates); ``p-body``/``stress_granule`` ->
``Cytoplasm`` (cytoplasmic granules). DeepLoc has no "condensate" class, so cytoplasmic granules are
better captured by the LLPS predictor (see :mod:`extras.eval.llps`).
"""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import tempfile

import numpy as np

from extras.eval._tools import deeploc_venv, torch_cache

# DeepLoc 2.1 organelle classes (the per-class probability columns of results_*.csv).
COMPARTMENTS: tuple[str, ...] = (
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
    "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome",
)


def deeploc_score(
    seqs: list[str], *, model: str = "Fast", device: str = "cuda", deeploc: str | None = None
) -> dict[str, np.ndarray]:
    """Per-sequence DeepLoc probabilities: ``{compartment: np.ndarray[n_seqs]}`` (order preserved).

    Empty sequences are dropped before scoring (DeepLoc errors on them); the returned arrays cover
    the non-empty inputs in order. Raises ``CalledProcessError`` if the DeepLoc run fails.
    """
    kept = [s for s in seqs if s]
    if not kept:
        return {c: np.empty(0) for c in COMPARTMENTS}

    exe = deeploc or deeploc_venv()
    env = {**os.environ, "TORCH_HOME": torch_cache()}  # cached ESM1b -> no download
    with tempfile.TemporaryDirectory() as td:
        fasta = os.path.join(td, "in.fasta")
        with open(fasta, "w") as fh:
            for i, s in enumerate(kept):
                fh.write(f">seq_{i}\n{s}\n")
        subprocess.run([exe, "-f", fasta, "-o", td, "-m", model, "-d", device],
                       check=True, env=env, stdout=subprocess.DEVNULL)
        res = sorted(glob.glob(os.path.join(td, "results_*.csv")))
        if not res:
            raise RuntimeError(f"DeepLoc produced no results_*.csv in {td}")
        rows = list(csv.DictReader(open(res[-1])))

    # DeepLoc preserves input order and our headers sort lexically the same as written; map by index.
    by_id = {r["Protein_ID"]: r for r in rows}
    out = {c: np.full(len(kept), np.nan) for c in COMPARTMENTS}
    for i in range(len(kept)):
        r = by_id.get(f"seq_{i}")
        if r is None:
            continue
        for c in COMPARTMENTS:
            out[c][i] = float(r[c])
    return out


def localization_stats(scores: dict[str, np.ndarray], *, expected: str | None = None) -> dict:
    """Summary over :func:`deeploc_score` output: mean P per class, plus (if ``expected`` given) the
    mean P(expected) and the fraction of sequences whose argmax compartment is ``expected``."""
    n = len(next(iter(scores.values()))) if scores else 0
    means = {c: float(np.nanmean(scores[c])) if n else float("nan") for c in COMPARTMENTS}
    out: dict = {"n": n, "mean_prob": means}
    if expected is not None and n:
        mat = np.stack([scores[c] for c in COMPARTMENTS], axis=1)  # [n, n_classes]
        argmax = np.array(COMPARTMENTS)[np.nanargmax(mat, axis=1)]
        out["expected"] = expected
        out["mean_expected"] = means.get(expected, float("nan"))
        out["frac_argmax_expected"] = float((argmax == expected).mean())
    return out
