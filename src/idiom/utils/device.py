"""Device selection from an explicit argument, IDIOM_DEVICE, or CUDA availability."""

from __future__ import annotations

import os

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device, or use IDIOM_DEVICE then CUDA/CPU for None or "auto"."""
    if device is not None and str(device) != "auto":
        return torch.device(device)  # explicit request wins
    env = os.environ.get("IDIOM_DEVICE")
    if env:
        return torch.device(env)  # env override (tests set IDIOM_DEVICE=cpu)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU-first fallback


def is_cpu_only(device: str | torch.device | None = None) -> bool:
    """Return whether resolve_device(device) selects CPU."""
    return resolve_device(device).type == "cpu"
