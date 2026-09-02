"""RL-SAE rewards: reward a policy for reproducing a target's interpretable SAE feature code.

Registers sae_only_<name> for every signature in the targets file -- the fraction of that target's
features that fire (are in the SAE top-k at any IDR residue) in the completion. Enable via the
rl_sae config block (this file is imported automatically):

    idiom_grpo init_from=... reward.rl_sae.enabled=true reward.rl_sae.signature=nucleolus

Keep the grpo.yaml entropy term on as the naturalness guardrail. Bring your own signature by pointing
IDIOM_SAEREWARD_FEATURES at a JSON of the same shape (build one with examples/python/05_feature_enrichment.py).

Environment variables:
    IDIOM_SAEREWARD_SAE       SAE to use as the lens (HF repo id or local dir)
    IDIOM_SAEREWARD_FEATURES  targets JSON, {case: {name: [feature ids]}}
    IDIOM_SAEREWARD_CASE      which case to use: "top30" (default) or "private30"
    IDIOM_SAEREWARD_DEVICE    torch device for the lens (default cuda if available)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from idiom.train.grpo.reward.base import register_reward

# Targets ship as user-editable data in the repo's top-level rewards/rl_sae_targets/ (run from the
# repo root); point IDIOM_SAEREWARD_FEATURES at your own JSON of the same shape to override.
_SAE_DIR = os.environ.get("IDIOM_SAEREWARD_SAE", "jxliu2/idiomsae-300M-L18-k32")
_FEATURES = os.environ.get(
    "IDIOM_SAEREWARD_FEATURES", "rewards/rl_sae_targets/idiomsae-300M-L18-k32.json")
_CASE = os.environ.get("IDIOM_SAEREWARD_CASE", "top30")


def _saedev() -> str:
    d = os.environ.get("IDIOM_SAEREWARD_DEVICE")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _sae():
    """Load the frozen base plus SAE lens once, sharing the policy's GPU."""
    from idiom import IDiomSAE
    return IDiomSAE.from_pretrained(_SAE_DIR, device=_saedev())


@lru_cache(maxsize=1)
def _featuresets() -> dict:
    """Return {name: [feature ids]} for the configured case of the targets file."""
    blob = json.loads(Path(_FEATURES).read_text())
    if _CASE not in blob:
        cases = [k for k in blob if not k.startswith("_")]
        raise KeyError(f"case {_CASE!r} not in {_FEATURES} (available: {cases}); "
                       f"set IDIOM_SAEREWARD_CASE")
    return blob[_CASE]


@lru_cache(maxsize=32)
def _target_ids(name: str):
    """Return the feature-id LongTensor for a signature, on the SAE device."""
    return torch.tensor(_featuresets()[name], device=_sae().device, dtype=torch.long)


@torch.no_grad()
def feature_match(idr: str, name: str) -> float:
    """Return the fraction of a signature's features that fire (top-k at any IDR residue).

    Args:
        idr (str): The decoded IDR residue string.
        name (str): Signature name in the targets file.

    Returns:
        float: Fraction of the signature's features that fire, or 0.0 for an empty string.
    """
    if not idr:
        return 0.0
    from idiom.data.fim import fim_unprompted
    from idiom.model.activations import extract_activations
    sae = _sae()
    s = fim_unprompted(idr, 0, len(idr))                            # "132" + idr (unprompted)
    tokens = torch.tensor([[sae.tok.start_id, *sae.tok.encode(s)]], device=sae.device)
    acts = extract_activations(sae.model, tokens, [sae.layer], tokenizer=sae.tok,
                               drop_markers=True, region=sae.region)[sae.layer]
    feats = sae.sae.encode_dense(acts.values.to(sae.device))        # [n_idr_res, num_latents]
    ids = _target_ids(name)
    fired = (feats[:, ids] > 0).any(dim=0).float()                  # [n_ids]
    return float(fired.mean())


def _sae_only_reward(name: str):
    """Build the per-sequence feature-match reward for a signature (no classifier in the loop)."""

    def reward(idr: str) -> float:
        return feature_match(idr, name) if idr else 0.0

    return reward


def _signature_names() -> list[str]:
    """Signature names in the configured targets file (falls back to none if unreadable)."""
    try:
        return sorted(_featuresets())
    except Exception:  # noqa: BLE001 - a bad/missing targets file must not break import
        return []


# Register a reward for every signature the targets file defines, so a user-supplied signature file
# gets rewards with no code change.
for _name in _signature_names():
    register_reward(f"sae_only_{_name}")(_sae_only_reward(_name))
