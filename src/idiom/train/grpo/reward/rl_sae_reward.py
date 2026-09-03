"""Rewards for reproducing a target's SAE feature code.

Importing this module registers one reward per signature in the targets file, named
"sae_only_<signature>". Each scores an IDR by the fraction of that signature's features that fire,
meaning they appear in the SAE top-k at any of the IDR's residues. The raw value is already a
fraction in [0, 1], so a term usually leaves it unshaped. Naming the module on the term imports it,
which is what keeps torch and the SAE out of a run that does not use one:

    reward.terms:
      - {reward: sae_only_nucleolus, module: idiom.train.grpo.reward.rl_sae_reward, weight: 1.0}

The targets file maps a case name to a mapping of signature name to feature ids, and can be built
with examples/scripts/feature_enrichment.py.

Environment variables:
    IDIOM_SAEREWARD_SAE: SAE to use as the lens, as a Hub repo id or local directory.
    IDIOM_SAEREWARD_FEATURES: path to the targets JSON, {case: {name: [feature ids]}}.
    IDIOM_SAEREWARD_CASE: which case of the targets file to read; "top30" by default.
    IDIOM_SAEREWARD_DEVICE: torch device for the lens; cuda when available by default.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from idiom import IDiomSAE
from idiom.data.fim import fim_unprompted
from idiom.model.activations import extract_activations
from idiom.train.grpo.reward.registry import register_reward

# Targets ship as user-editable data in the repo's top-level rewards/rl_sae_targets/ (run from the
# repo root); point IDIOM_SAEREWARD_FEATURES at your own JSON of the same shape to override.
_SAE_DIR = os.environ.get("IDIOM_SAEREWARD_SAE", "jxliu2/idiomsae-300M-L18-k32")
_FEATURES = os.environ.get(
    "IDIOM_SAEREWARD_FEATURES", "rewards/rl_sae_targets/idiomsae-300M-L18-k32.json")
_CASE = os.environ.get("IDIOM_SAEREWARD_CASE", "top30")


def _saedev() -> str:
    """Return the torch device for the lens, from IDIOM_SAEREWARD_DEVICE or availability."""
    d = os.environ.get("IDIOM_SAEREWARD_DEVICE")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _sae():
    """Load and cache the SAE and its host model."""
    return IDiomSAE.from_pretrained(_SAE_DIR, device=_saedev())


@lru_cache(maxsize=1)
def _featuresets() -> dict:
    """Return the {name: [feature ids]} mapping for the configured case of the targets file.

    Raises:
        KeyError: If the configured case is not present in the targets file.
    """
    blob = json.loads(Path(_FEATURES).read_text())
    if _CASE not in blob:
        cases = [k for k in blob if not k.startswith("_")]
        raise KeyError(f"case {_CASE!r} not in {_FEATURES} (available: {cases}); "
                       f"set IDIOM_SAEREWARD_CASE")
    return blob[_CASE]


@lru_cache(maxsize=32)
def _target_ids(name: str):
    """Return a signature's feature ids as a LongTensor on the SAE's device."""
    return torch.tensor(_featuresets()[name], device=_sae().device, dtype=torch.long)


@torch.no_grad()
def feature_match(idr: str, name: str) -> float:
    """Return the fraction of a signature's features that fire on an IDR.

    The IDR is encoded through the SAE in the unprompted FIM format, and a feature fires if it is
    in the SAE top-k at any of the IDR's residues.

    Args:
        idr (str): The decoded IDR residue string.
        name (str): Signature name in the targets file.

    Returns:
        float: The fraction of the signature's features that fire, or 0.0 for an empty string.

    Raises:
        KeyError: If name is not a signature in the configured case.
    """
    if not idr:
        return 0.0
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
    """Build the per-idr feature-match reward for one signature."""

    def reward(idr: str) -> float:
        return feature_match(idr, name) if idr else 0.0

    return reward


def _signature_names() -> list[str]:
    """Return the sorted signature names in the configured targets file, or [] if it is unreadable."""
    try:
        return sorted(_featuresets())
    except Exception:  # a bad/missing targets file must not break import
        return []


# Register a reward for every signature the targets file defines, so a user-supplied signature file
# gets rewards with no code change.
for _name in _signature_names():
    register_reward(f"sae_only_{_name}")(_sae_only_reward(_name))
