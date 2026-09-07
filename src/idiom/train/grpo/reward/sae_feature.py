"""Reward the fraction of a target SAE signature active in an IDR.

Signature JSON maps case names to {signature_name: [feature_ids]}.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

from idiom import IDiomSAE
from idiom.data.fim import fim_unprompted
from idiom.model.activations import extract_activations
from idiom.train.grpo.reward.resolve import Reward, batchify

DEFAULT_SAE = "jxliu2/idiomsae-300M-L18-k32"
DEFAULT_FEATURES = str(Path(__file__).resolve().parent / "sae_signatures.json")
DEFAULT_CASE = "top30"


@lru_cache(maxsize=4)
def _sae(sae_dir: str, device: str | None):
    """Load and cache the SAE and its host model."""
    return IDiomSAE.from_pretrained(
        sae_dir, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))


@lru_cache(maxsize=8)
def _featuresets(features: str, case: str) -> dict:
    """Read a case as {signature_name: [feature_ids]}; raise KeyError if absent."""
    blob = json.loads(Path(features).read_text())
    if case not in blob:
        cases = [k for k in blob if not k.startswith("_")]
        raise KeyError(f"case {case!r} not in {features} (available: {cases}); set the term's "
                       f"case")
    return blob[case]


@lru_cache(maxsize=32)
def _target_ids(signature: str, features: str, case: str, sae_dir: str, device: str | None):
    """Return signature indices on the SAE device; raise KeyError for an unknown signature."""
    sets = _featuresets(features, case)
    if signature not in sets:
        raise KeyError(f"signature {signature!r} not in case {case!r} of {features} "
                       f"(available: {sorted(sets)}); set the term's signature")
    return torch.tensor(sets[signature], device=_sae(sae_dir, device).device, dtype=torch.long)


@torch.no_grad()
def feature_match(idr: str, signature: str, *, features: str = DEFAULT_FEATURES,
                  case: str = DEFAULT_CASE, sae: str = DEFAULT_SAE,
                  device: str | None = None) -> float:
    """Return the fraction of signature features with positive activation on an IDR.

    Encode in unprompted FIM format; a feature counts if active at any selected residue.

    Args:
        idr: The decoded IDR residue string.
        signature: Signature name in the signature file.
        features: Path to the signature JSON.
        case: Which case of the signature file to read.
        sae: SAE to use as the lens, as a Hub repo id or a local directory.
        device: Torch device for the lens; cuda when available if None.

    Returns:
        The fraction of the signature's features that fire, or 0.0 for an empty string.

    Raises:
        KeyError: If the case or the signature is not in the signature file.
    """
    if not idr:
        return 0.0
    lens = _sae(sae, device)
    s = fim_unprompted(idr, 0, len(idr))
    tokens = torch.tensor([[lens.tok.start_id, *lens.tok.encode(s)]], device=lens.device)
    acts = extract_activations(lens.model, tokens, [lens.layer], tokenizer=lens.tok,
                               drop_markers=True, region=lens.region)[lens.layer]
    feats = lens.sae.encode_dense(acts.values.to(lens.device))
    ids = _target_ids(signature, features, case, sae, device)
    fired = (feats[:, ids] > 0).any(dim=0).float()
    return float(fired.mean())


def sae_signature(signature: str, *, features: str = DEFAULT_FEATURES, case: str = DEFAULT_CASE,
                  sae: str = DEFAULT_SAE, device: str | None = None) -> Reward:
    """Build a feature-match reward, validating the signature immediately.

    Load the SAE lazily on the first non-empty IDR.

    Args:
        signature: Signature name in the signature file.
        features: Path to the signature JSON, {case: {name: [feature ids]}}.
        case: Which case of the signature file to read.
        sae: SAE to use as the lens, as a Hub repo id or a local directory.
        device: Torch device for the lens; cuda when available if None.

    Returns:
        A raw reward in [0, 1] per IDR.

    Raises:
        ValueError: If the signature file cannot be read, or holds neither the case nor the
            signature.
    """
    try:
        sets = _featuresets(features, case)
    except Exception as e:
        raise ValueError(
            f"sae_signature: cannot read case {case!r} of {features!r} ({type(e).__name__}: {e}). "
            f"Build a signature with cookbook/notebooks/feature_enrichment.ipynb, then point the "
            f"term's features at it.") from e
    if signature not in sets:
        raise ValueError(f"sae_signature: no signature {signature!r} in case {case!r} of "
                         f"{features!r}; available: {sorted(sets)}")

    return batchify(lambda idr: feature_match(idr, signature, features=features, case=case, sae=sae,
                                          device=device))
