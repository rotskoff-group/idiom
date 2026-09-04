"""Device resolution.

A device is taken from an explicit argument, then the IDIOM_DEVICE environment variable, then
"cuda" if a GPU is visible, else "cpu".
"""

from __future__ import annotations

import os

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device.

    Args:
        device (str | torch.device | None): An explicit device, or None or "auto" to fall back to
            IDIOM_DEVICE and then to a GPU if one is visible.

    Returns:
        torch.device: The resolved device.
    """
    if device is not None and str(device) != "auto":
        return torch.device(device)  # explicit request wins
    env = os.environ.get("IDIOM_DEVICE")
    if env:
        return torch.device(env)  # env override (tests set IDIOM_DEVICE=cpu)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU-first fallback


def is_cpu_only(device: str | torch.device | None = None) -> bool:
    """Return whether the resolved device is CPU.

    Args:
        device (str | torch.device | None): Device to resolve; see resolve_device.

    Returns:
        bool: True if the resolved device is CPU.
    """
    return resolve_device(device).type == "cpu"
