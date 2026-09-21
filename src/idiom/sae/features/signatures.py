"""Read, validate, combine, and write feature targets in the GRPO JSON format."""

import json
from pathlib import Path

import numpy as np


def validate_signature(ids, *, num_latents: int | None = None) -> list[int]:
    """Require nonempty, unique, nonnegative integer feature IDs, optionally bounded."""
    ids = list(ids)
    if not ids or any(
        isinstance(f, bool)
        or not isinstance(f, (int, np.integer))
        or f < 0
        or (num_latents is not None and f >= num_latents)
        for f in ids
    ):
        raise ValueError("A signature needs nonnegative integer IDs within the SAE latent range")
    if len(set(ids)) != len(ids):
        raise ValueError("A signature must not contain duplicate feature IDs")
    return [int(f) for f in ids]


def load_signatures(path, *, case="top30", sae=None, num_latents=None) -> dict[str, list[int]]:
    """Load a case, checking declared SAE identity when supplied.

    Legacy files without SAE provenance remain readable. Identity compares the saved
    Hub ID or path; it is not a checksum of weights. Keep model revisions with run records.
    """
    blob = json.loads(Path(path).read_text())
    recorded = blob.get("_provenance", {}).get("sae")
    if sae is not None and recorded is not None and str(sae) != recorded:
        raise ValueError(f"Signature SAE {recorded!r} does not match requested SAE {str(sae)!r}")
    if case not in blob:
        raise KeyError(f"Signature case {case!r} not found in {path}")
    if not isinstance(blob[case], dict):
        raise ValueError("A signature case must map names to feature ID lists")
    return {name: validate_signature(ids, num_latents=num_latents) for name, ids in blob[case].items()}


def combine_signatures(signatures: dict[str, list[int]]) -> list[int]:
    """Union component signatures in insertion order; callers retain components for evaluation."""
    if not signatures:
        raise ValueError("At least one component signature is required")
    return list(dict.fromkeys(f for ids in signatures.values() for f in validate_signature(ids)))


def write_signature(path, signatures, *, case="top30", provenance=None) -> Path:
    """Update one case while preserving other cases and compatible file-level provenance."""
    if not case.strip() or case.startswith("_") or any(not name.strip() for name in signatures):
        raise ValueError("Provide nonempty signature names and a non-reserved case name")
    validated = {name: validate_signature(ids) for name, ids in signatures.items()}
    out = Path(path)
    blob = json.loads(out.read_text()) if out.exists() else {}
    old_sae = blob.get("_provenance", {}).get("sae")
    new_sae = (provenance or {}).get("sae")
    if old_sae and new_sae and old_sae != new_sae:
        raise ValueError("Cannot mix signatures from different SAEs in one file")
    blob[case] = validated
    if provenance:
        blob.setdefault("_provenance", {}).update(provenance)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    return out
