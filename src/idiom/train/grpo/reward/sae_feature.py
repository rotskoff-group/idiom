"""Reward the fraction of a target SAE signature active in an IDR.

Signature JSON maps case names to {signature_name: [feature_ids]}.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from idiom import IDiomSAE
from idiom.sae.features.signatures import load_signatures
from idiom.train.grpo.reward.resolve import Reward, batchify

DEFAULT_SAE = "jxliu2/idiomsae-300M-L18-k32"
DEFAULT_FEATURES = str(Path(__file__).resolve().parent / "sae_signatures.json")
DEFAULT_CASE = "top30"


@lru_cache(maxsize=4)
def _sae(sae_dir: str, device: str | None) -> IDiomSAE:
    """Load and cache the SAE and its host model."""
    return IDiomSAE.from_pretrained(
        sae_dir, device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    )


@lru_cache(maxsize=8)
def _featuresets(features: str, case: str) -> dict:
    """Read a case as {signature_name: [feature_ids]}; raise KeyError if absent."""
    return load_signatures(features, case=case)


@lru_cache(maxsize=32)
def _target_ids(signature: str, features: str, case: str, sae_dir: str, device: str | None) -> torch.Tensor:
    """Return signature indices on the SAE device; raise KeyError for an unknown signature."""
    lens = _sae(sae_dir, device)
    sets = load_signatures(features, case=case, sae=sae_dir, num_latents=lens.sae.num_latents)
    if signature not in sets:
        raise KeyError(
            f"signature {signature!r} not in case {case!r} of {features} "
            f"(available: {sorted(sets)}); set the term's signature"
        )
    return torch.tensor(sets[signature], device=lens.device, dtype=torch.long)


@torch.no_grad()
def signature_presence(
    idrs: list[str],
    signature: str,
    *,
    features: str = DEFAULT_FEATURES,
    case: str = DEFAULT_CASE,
    sae: str = DEFAULT_SAE,
    device: str | None = None,
) -> np.ndarray:
    """Return [sequences, target features] presence using the reward's frozen lens.

    Column order follows the signature. Empty sequences have no active features.
    This is the same positive-anywhere criterion used by feature_match.
    """
    lens = _sae(sae, device)
    ids = _target_ids(signature, features, case, sae, device).cpu().numpy()
    presence = np.zeros((len(idrs), len(ids)), dtype=bool)
    for i, sequence in enumerate(idrs):
        if sequence:
            peaks, _ = lens.encode(sequence, pool="max")
            presence[i] = peaks[0, ids] > 0
    return presence


@torch.no_grad()
def feature_match(
    idr: str,
    signature: str,
    *,
    features: str = DEFAULT_FEATURES,
    case: str = DEFAULT_CASE,
    sae: str = DEFAULT_SAE,
    device: str | None = None,
) -> float:
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
    return float(
        signature_presence([idr], signature, features=features, case=case, sae=sae, device=device).mean()
    )


def sae_signature(
    signature: str,
    *,
    features: str = DEFAULT_FEATURES,
    case: str = DEFAULT_CASE,
    sae: str = DEFAULT_SAE,
    device: str | None = None,
) -> Reward:
    """Build a feature-match reward, validating the signature immediately.

    Load the SAE lazily on the first non-empty IDR.

    Args:
        signature: Signature name in the signature file.
        features: Path to the signature JSON, {case: {name: [feature ids]}}.
        case: Which case of the signature file to read.
        sae: SAE to use as the lens, as a Hub repo id or a local directory.
        device: Torch device for the lens; cuda when available if None.

    Returns:
        A batch-scoring callable returning one feature-match fraction in [0, 1]
        per IDR, in input order. Empty strings score 0.

    Raises:
        ValueError: If the signature file cannot be read, or holds neither the case nor the
            signature.
    """
    try:
        sets = load_signatures(features, case=case, sae=sae)
    except Exception as e:
        raise ValueError(
            f"sae_signature: cannot read case {case!r} of {features!r} ({type(e).__name__}: {e}). "
            f"Build a signature with cookbook/notebooks/04_discover_feature_signature.ipynb, then point the "
            f"term's features at it."
        ) from e
    if signature not in sets:
        raise ValueError(
            f"sae_signature: no signature {signature!r} in case {case!r} of "
            f"{features!r}; available: {sorted(sets)}"
        )

    return batchify(
        lambda idr: feature_match(idr, signature, features=features, case=case, sae=sae, device=device)
    )
