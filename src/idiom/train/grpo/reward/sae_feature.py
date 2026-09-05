"""A reward for reproducing a target's SAE feature code.

sae_signature is a reward factory like any other, so which signature a run chases, and which SAE
and signature file it is read from, are arguments in the term and are saved with the run.

    reward.terms:
      - {reward: {name: sae_signature, signature: nucleolus, features: signature.json},
         label: sae, weight: 1.0}

The reward scores an IDR by the fraction of the signature's features that appear in the SAE top-k
at any of the IDR's residues, a value in [0, 1]. The signature file maps a case name to a mapping
of signature name to feature ids, and can be built with cookbook/notebooks/feature_enrichment.ipynb.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

from idiom import IDiomSAE
from idiom.data.fim import fim_unprompted
from idiom.model.activations import extract_activations
from idiom.train.grpo.reward.resolve import Reward, lift

# The signatures for the released SAE ship next to this module, so the default works in any install
# and from any working directory; point a term's features at your own JSON of the same shape
# (cookbook/notebooks/feature_enrichment.ipynb writes one) to override.
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
    """Return the {name: [feature ids]} mapping for one case of a signature file.

    Args:
        features (str): Path to the signature JSON, {case: {name: [feature ids]}}.
        case (str): Which case of the file to read.

    Returns:
        dict: The signature name to feature id list mapping for that case.

    Raises:
        KeyError: If the case is not present in the signature file.
    """
    blob = json.loads(Path(features).read_text())
    if case not in blob:
        cases = [k for k in blob if not k.startswith("_")]
        raise KeyError(f"case {case!r} not in {features} (available: {cases}); set the term's "
                       f"case")
    return blob[case]


@lru_cache(maxsize=32)
def _target_ids(signature: str, features: str, case: str, sae_dir: str, device: str | None):
    """Return a signature's feature ids as a LongTensor on the SAE's device.

    Raises:
        KeyError: If the signature is not in the given case of the signature file.
    """
    sets = _featuresets(features, case)
    if signature not in sets:
        raise KeyError(f"signature {signature!r} not in case {case!r} of {features} "
                       f"(available: {sorted(sets)}); set the term's signature")
    return torch.tensor(sets[signature], device=_sae(sae_dir, device).device, dtype=torch.long)


@torch.no_grad()
def feature_match(idr: str, signature: str, *, features: str = DEFAULT_FEATURES,
                  case: str = DEFAULT_CASE, sae: str = DEFAULT_SAE,
                  device: str | None = None) -> float:
    """Return the fraction of a signature's features that fire on an IDR.

    The IDR is encoded through the SAE in the unprompted FIM format; a feature fires if it is in
    the SAE top-k at any of the IDR's residues.

    Args:
        idr (str): The decoded IDR residue string.
        signature (str): Signature name in the signature file.
        features (str): Path to the signature JSON.
        case (str): Which case of the signature file to read.
        sae (str): SAE to use as the lens, as a Hub repo id or a local directory.
        device (str | None): Torch device for the lens; cuda when available if None.

    Returns:
        float: The fraction of the signature's features that fire, or 0.0 for an empty string.

    Raises:
        KeyError: If the case or the signature is not in the signature file.
    """
    if not idr:
        return 0.0
    lens = _sae(sae, device)
    s = fim_unprompted(idr, 0, len(idr))                            # "132" + idr (unprompted)
    tokens = torch.tensor([[lens.tok.start_id, *lens.tok.encode(s)]], device=lens.device)
    acts = extract_activations(lens.model, tokens, [lens.layer], tokenizer=lens.tok,
                               drop_markers=True, region=lens.region)[lens.layer]
    feats = lens.sae.encode_dense(acts.values.to(lens.device))      # [n_idr_res, num_latents]
    ids = _target_ids(signature, features, case, sae, device)
    fired = (feats[:, ids] > 0).any(dim=0).float()                  # [n_ids]
    return float(fired.mean())


def sae_signature(signature: str, *, features: str = DEFAULT_FEATURES, case: str = DEFAULT_CASE,
                  sae: str = DEFAULT_SAE, device: str | None = None) -> Reward:
    """Build the feature-match reward for one signature, from a term's arguments.

    The signature file is read and validated here, so a missing file, case or signature fails at
    config time rather than on the first training step. The SAE itself is loaded lazily, on the
    first IDR scored.

    Args:
        signature (str): Signature name in the signature file.
        features (str): Path to the signature JSON, {case: {name: [feature ids]}}.
        case (str): Which case of the signature file to read.
        sae (str): SAE to use as the lens, as a Hub repo id or a local directory.
        device (str | None): Torch device for the lens; cuda when available if None.

    Returns:
        Reward: A raw reward in [0, 1] per IDR.

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

    return lift(lambda idr: feature_match(idr, signature, features=features, case=case, sae=sae,
                                          device=device))
