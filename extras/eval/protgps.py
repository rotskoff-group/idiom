"""ProtGPS condensate scoring for *evaluation* (distinct from the GRPO *reward* in
``rewards/protgps_reward.py``, which this loads). Used to (a) score generations against every
compartment and (b) build the target-selectivity ("specificity") matrix that asks whether each GRPO
model is elevated on *its own* target compartment rather than uniformly raising all condensate scores.

ProtGPS is the same classifier the reward uses (small ESM-2 + a 12-compartment head), so this is an
*in-distribution* readout, not an independent validator — pair it with the orthogonal
DeepLoc/catGRANULE checks. GPU by default; honors ``$IDIOM_PROTGPS_DEVICE``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]  # extras/eval/protgps.py -> repo root
_REWARD_PY = _REPO / "rewards" / "protgps_reward.py"


def _load_reward_module():
    """Import ``rewards/protgps_reward.py`` by file path (rewards/ isn't an installed package)."""
    spec = importlib.util.spec_from_file_location("protgps_reward", _REWARD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_compartments() -> list[str]:
    return list(_load_reward_module().COMPARTMENTS)


@torch.no_grad()
def protgps_score(seqs: list[str], *, batch_size: int = 64) -> np.ndarray:
    """Per-sequence ProtGPS probabilities, shape ``[n_seqs, 12]`` (order preserved; empties -> NaN row)."""
    mod = _load_reward_module()
    model = mod._load_model()
    n_c = len(mod.COMPARTMENTS)
    out = np.full((len(seqs), n_c), np.nan)
    idx = [i for i, s in enumerate(seqs) if s]
    for b in range(0, len(idx), batch_size):
        chunk = idx[b:b + batch_size]
        logit = model.model({"x": [seqs[i][:mod._MAX_LEN] for i in chunk]})["logit"]
        probs = torch.sigmoid(logit).cpu().numpy()
        for row, i in zip(probs, chunk):
            out[i] = row
    return out


def protgps_stats(scores: np.ndarray, *, compartment: str) -> dict:
    """Mean P(``compartment``) over the scored sequences."""
    comps = get_compartments()
    j = comps.index(compartment)
    col = scores[:, j]
    col = col[~np.isnan(col)]
    return {"n": int(col.size), "compartment": compartment,
            "mean": float(col.mean()) if col.size else float("nan")}


def specificity_matrix(scores_by_model: dict[str, np.ndarray], *, base: str = "base") -> dict:
    """Model x compartment mean-probability matrix + target-selectivity readout.

    ``scores_by_model`` maps a model label -> its ``protgps_score`` array ``[n, 12]``. Returns the
    mean-P matrix, the delta vs the ``base`` model, and per-model whether its own-named compartment is
    the argmax (diagonal dominance => the reward is target-selective, not a generic condensate push).
    """
    comps = get_compartments()
    means = {m: np.nanmean(s, axis=0) for m, s in scores_by_model.items()}
    base_mean = means.get(base)
    out: dict = {"compartments": comps, "means": {m: v.tolist() for m, v in means.items()}}
    if base_mean is not None:
        out["delta_vs_base"] = {m: (v - base_mean).tolist() for m, v in means.items()}
    own = {}
    for m, v in means.items():
        if m in comps:
            argmax = comps[int(np.nanargmax(v))]
            own[m] = {"own_mean": float(v[comps.index(m)]), "argmax": argmax,
                      "own_is_argmax": bool(argmax == m)}
    out["own_target"] = own
    return out
