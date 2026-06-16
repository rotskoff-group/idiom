"""ProtGPS localization reward for GRPO — ``f(idr: str) -> float``.

ProtGPS is a vendored condensate-localization classifier (``rewards/protgps/``) on top of a
small ESM-2 (esm2_t6_8M, pulled from the local torch.hub cache — no ``fair-esm`` install
needed). It maps a residue string to 12 compartment probabilities (sigmoid). We register one
reward per compartment plus ``protgps_max`` / ``protgps_mean``; ``protgps`` aliases the max.

Use it::

    idiom_grpo init_from=... reward.module=rewards/protgps_reward.py reward.name=protgps_nucleolus

Paths/devices (override via env):
    IDIOM_PROTGPS_DIR     parent dir holding protgps/*.ckpt + esm_models/esm2
    IDIOM_PROTGPS_DEVICE  torch device for the scorer (default: cuda if available else cpu;
                          under srun, cuda == the allocated GPU since CUDA_VISIBLE_DEVICES is set)

Caveat (see steering findings): ProtGPS rewards compositional extremity — high scores are
easy to reach with low-complexity tracts. Pair it with the entropy term in grpo.yaml
(target_entropy ~2.7 nats) to keep completions natural and avoid reward-hacking.
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
    """Load + cache the vendored ProtGPS classifier (once per process)."""
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
    """12 compartment probabilities for one IDR; zeros for an empty/over-long-truncated string."""
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


@register_reward("protgps_max")
def protgps_max(idr: str) -> float:
    """Max probability across all 12 compartments (generic "is it a condensate IDR")."""
    return float(protgps_scores(idr).max()) if idr else 0.0


@register_reward("protgps_mean")
def protgps_mean(idr: str) -> float:
    return float(protgps_scores(idr).mean()) if idr else 0.0


# Default alias so the stock grpo.yaml (reward.name: protgps) resolves.
register_reward("protgps")(protgps_max)
