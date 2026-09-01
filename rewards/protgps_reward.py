"""ProtGPS localization reward for GRPO with the contract f(idr: str) -> float.

ProtGPS is a vendored condensate-localization classifier (rewards/protgps/) on top of a small ESM-2
(esm2_t6_8M, pulled from the local torch.hub cache, so no fair-esm install is needed). It maps a
residue string to 12 compartment probabilities (sigmoid). We register one reward per compartment plus
protgps_max and protgps_mean; protgps aliases the max.

Use it:

    idiom_grpo init_from=... reward.module=rewards/protgps_reward.py reward.name=protgps_nucleolus

Paths and devices are overridable via environment variables. IDIOM_PROTGPS_DIR is the parent dir
holding protgps/*.ckpt and esm_models/esm2. IDIOM_PROTGPS_DEVICE is the torch device for the scorer
(default: cuda if available else cpu; under srun, cuda is the allocated GPU since
CUDA_VISIBLE_DEVICES is set).

Caveat: ProtGPS rewards compositional extremity — high scores are easy to reach with low-complexity
tracts. Pair it with the entropy term in grpo.yaml (target_entropy about 3.68 bits = 2.55 nats) to
keep completions natural and avoid reward-hacking.
"""

from __future__ import annotations

import os
import pickle
from argparse import Namespace
from functools import lru_cache
from pathlib import Path

import torch

from idiom.train.grpo.rewards import register_reward

COMPARTMENTS = [
    "nuclear_speckle", "p-body", "pml-bdoy", "post_synaptic_density", "stress_granule",
    "chromosome", "nucleolus", "nuclear_pore_complex", "cajal_body", "rna_granule",
    "cell_junction", "transcriptional",
]

_DEFAULT_DIR = "/data2/scratch/group_scratch/idr_plm/2026-06-14_refactor/models/protgps"
_CKPT_STEM = "protgps/32bf44b16a4e770a674896b81dfb3729"
_MAX_LEN = 1800  # ProtGPS sequence-length ceiling


def _device() -> str:
    d = os.environ.get("IDIOM_PROTGPS_DEVICE")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the vendored ProtGPS classifier (once per process)."""
    import sys

    parent = Path(os.environ.get("IDIOM_PROTGPS_DIR", _DEFAULT_DIR))
    pkg_dir = str(Path(__file__).resolve().parent / "protgps")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    from protgps.utils.loading import get_object  # noqa: PLC0415

    args = Namespace(**pickle.load(open(parent / f"{_CKPT_STEM}.args", "rb")))
    args.model_path = str(parent / f"{_CKPT_STEM}epoch=26.ckpt")
    args.pretrained_hub_dir = str(parent / "esm_models/esm2")  # torch.hub cache (offline)

    model = get_object(args.lightning_name, "lightning")(args)
    model = model.load_from_checkpoint(
        checkpoint_path=args.model_path,
        strict=not args.relax_checkpoint_matching,
        args=args,
    )
    return model.eval().to(_device())


def protgps_scores(idr: str) -> torch.Tensor:
    """Return the 12 compartment probabilities (sigmoid) for one IDR.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        torch.Tensor: A length-12 tensor of compartment probabilities, all zero for an empty string;
            strings longer than the length ceiling are truncated before scoring.
    """
    if not idr:
        return torch.zeros(len(COMPARTMENTS))
    model = _load_model()
    with torch.no_grad():
        out = model.model({"x": [idr[:_MAX_LEN]]})
        return torch.sigmoid(out["logit"]).squeeze(0).cpu()


def _compartment_reward(compartment: str):
    idx = COMPARTMENTS.index(compartment)
    return lambda idr: float(protgps_scores(idr)[idx]) if idr else 0.0


# Per-compartment rewards: protgps_<compartment> -> P(that compartment).
for _c in COMPARTMENTS:
    register_reward(f"protgps_{_c}")(_compartment_reward(_c))

# Clean alias: the ProtGPS class label carries a typo ("pml-bdoy"); dataset + enrichment use
# "pml_body", so expose protgps_pml_body -> the same classifier index.
register_reward("protgps_pml_body")(_compartment_reward("pml-bdoy"))


# Selective ("off-target-penalized") reward: 1 - MSE(P, one-hot target) over ALL 12 compartments.
# Maximized when P(target)->1 AND P(every other compartment)->0, so it directly penalizes off-target
# spillover (e.g. the p-body<->stress_granule cross-talk) that the plain P(target) reward ignores.
SELECTIVE_TARGETS = ["nucleolus", "chromosome", "stress_granule", "p-body"]


def _selective_reward(compartment: str):
    t = torch.zeros(len(COMPARTMENTS))
    t[COMPARTMENTS.index(compartment)] = 1.0
    return lambda idr: float(1.0 - ((protgps_scores(idr) - t) ** 2).mean()) if idr else 0.0


for _c in SELECTIVE_TARGETS:
    register_reward(f"protgps_sel_{_c}")(_selective_reward(_c))


# Base-anchored MSE variant: same 1 - MSE(P, target) as protgps_sel_*, but the off-target targets are
# the BASE model's natural levels (BASE_P) instead of 0. Penalizing deviation from BASE (rather than
# from 0) removes the reward for suppressing off-targets BELOW their baseline -- the incentive that
# drove the acidic reward-hack in protgps_sel_*, since several off-targets (chromosome 0.24, p-body
# 0.16, nucleolus 0.14) are naturally nonzero and forcing them to 0 required unnatural composition.
# Anchor = mean ProtGPS over base de-novo generations (05_generation .../protgps/base.csv, 2026-06-18).
BASE_P = {
    "nuclear_speckle": 0.0047, "p-body": 0.1644, "pml-bdoy": 0.0044, "post_synaptic_density": 0.0233,
    "stress_granule": 0.0477, "chromosome": 0.2384, "nucleolus": 0.1399, "nuclear_pore_complex": 0.0784,
    "cajal_body": 0.0107, "rna_granule": 0.0000, "cell_junction": 0.0921, "transcriptional": 0.0009,
}


def _anchor_reward(compartment: str):
    t = torch.tensor([BASE_P[c] for c in COMPARTMENTS])
    t[COMPARTMENTS.index(compartment)] = 1.0   # on-target -> 1; off-targets -> their base level
    return lambda idr: float(1.0 - ((protgps_scores(idr) - t) ** 2).mean()) if idr else 0.0


for _c in SELECTIVE_TARGETS:
    register_reward(f"protgps_anchor_{_c}")(_anchor_reward(_c))


@register_reward("protgps_max")
def protgps_max(idr: str) -> float:
    """Return the max probability across all 12 compartments (a generic condensate-IDR signal).

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: The largest compartment probability, or 0.0 for an empty string.
    """
    return float(protgps_scores(idr).max()) if idr else 0.0


@register_reward("protgps_mean")
def protgps_mean(idr: str) -> float:
    """Return the mean probability across all 12 compartments.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: The mean compartment probability, or 0.0 for an empty string.
    """
    return float(protgps_scores(idr).mean()) if idr else 0.0


# Default alias so the stock grpo.yaml (reward.name: protgps) resolves.
register_reward("protgps")(protgps_max)


# --- ProtGPS combined with the SAE feature code (RL-SAE + classifier) -------------------------
# These need BOTH ProtGPS and the SAE lens, so they live here rather than in rl_sae_reward.py:
# the dependency flows one way (protgps_reward -> rl_sae_reward), which keeps the pure-SAE
# rewards runnable without ProtGPS, its weights, or pytorch_lightning.
_LAMBDA = float(os.environ.get("IDIOM_SAEREWARD_LAMBDA", "0.5"))
_PC_ALIAS = {"pml_body": "pml-bdoy"}   # dataset spelling -> the ProtGPS class-label typo


def _sae_feature_match(idr: str, name: str) -> float:
    """Call feature_match from the sibling SAE reward module (loaded by path, like this one)."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rl_sae_reward import feature_match  # noqa: PLC0415
    return feature_match(idr, name)


def _feat_reward(name: str):
    """Build ProtGPS(target) + lambda * feature-match: classifier plus interpretable code."""
    idx = COMPARTMENTS.index(_PC_ALIAS.get(name, name))

    def reward(idr: str) -> float:
        if not idr:
            return 0.0
        return float(protgps_scores(idr)[idx]) + _LAMBDA * _sae_feature_match(idr, name)

    return reward


def _protgps_min_reward(a: str, b: str):
    """Build a chimera monitor: min of two compartments' ProtGPS heads (never in the reward)."""
    ia = COMPARTMENTS.index(_PC_ALIAS.get(a, a))
    ib = COMPARTMENTS.index(_PC_ALIAS.get(b, b))

    def reward(idr: str) -> float:
        if not idr:
            return 0.0
        s = protgps_scores(idr)
        return float(min(s[ia], s[ib]))

    return reward


def _sae_signature_names() -> list[str]:
    """Signature names available in the SAE targets file (empty if it cannot be read)."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from rl_sae_reward import _signature_names  # noqa: PLC0415
        return _signature_names()
    except Exception:  # noqa: BLE001 - missing/misconfigured targets must not break import
        return []


for _name in _sae_signature_names():
    if _PC_ALIAS.get(_name, _name) in COMPARTMENTS:
        register_reward(f"protgps_feat_{_name}")(_feat_reward(_name))
    if "__" in _name:                                   # chimera pair key "A__B"
        _a, _b = _name.split("__", 1)
        if all(_PC_ALIAS.get(x, x) in COMPARTMENTS for x in (_a, _b)):
            register_reward(f"protgps_min_{_name}")(_protgps_min_reward(_a, _b))
