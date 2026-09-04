"""Learning-rate schedules."""

from __future__ import annotations

import math

import torch


def warmup_cosine(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a schedule that warms up linearly, then decays on a cosine curve.

    The multiplier rises to 1.0 over warmup_steps, follows a cosine down to min_lr_ratio at
    max_steps, and stays there.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate is scheduled.
        warmup_steps (int): Number of linear warmup steps before the cosine decay begins.
        max_steps (int): Step at which the cosine decay reaches min_lr_ratio.
        min_lr_ratio (float): Floor of the decay, as a fraction of the base learning rate.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The configured scheduler, stepped per optimizer step.
    """

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)  # linear warmup
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
