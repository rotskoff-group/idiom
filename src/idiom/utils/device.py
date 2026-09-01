"""CPU-first device resolution.

This is a shared GPU box: every component must be runnable CPU-only on demand. Resolution order,
highest priority first:

1. an explicit device argument (anything other than None or "auto");
2. the IDIOM_DEVICE environment variable (tests set this to "cpu");
3. "cuda" if a GPU is visible, else "cpu".

Force CPU anywhere with IDIOM_DEVICE=cpu.
"""

from __future__ import annotations

import os

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device following the CPU-first order documented above.

    Args:
        device (str | torch.device | None): An explicit device, or None or "auto" to fall back to
            IDIOM_DEVICE and then a GPU if one is visible.

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
    """Return True when the resolved device is CPU (handy for skipping GPU-only paths).

    Args:
        device (str | torch.device | None): Device to resolve; see resolve_device.

    Returns:
        bool: True if the resolved device is CPU.
    """
    return resolve_device(device).type == "cpu"
