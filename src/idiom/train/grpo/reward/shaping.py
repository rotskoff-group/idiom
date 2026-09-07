"""Scalar reward shaping: identity, quadratic penalties, and Gaussian scores.

Custom shaping factories use the same "module:function" specs as rewards.
"""

from __future__ import annotations

import math
from collections.abc import Callable

Shaping = Callable[[float], float]


def tolerance(target: float, width: float) -> float:
    """Return abs(target) * width, or width when target is zero.

    Raises:
        ValueError: If width is not positive.
    """
    if width <= 0:
        raise ValueError(f"shaping width must be positive, got {width!r}")
    return abs(target) * width if target else width


def quadratic_penalty(value: float, target: float, width: float = 1.0) -> float:
    """Return -((value - target) / tolerance(target, width))**2.

    The score is 0 at the target and -1 one tolerance away. Width must be positive.
    """
    return -((value - target) / tolerance(target, width)) ** 2


def gaussian_score(value: float, target: float, width: float = 1.0) -> float:
    """Return exp(-((value - target) / tolerance(target, width))**2).

    The score is 1 at the target and approaches 0 away from it. Width must be positive.
    """
    return math.exp(-((value - target) / tolerance(target, width)) ** 2)


def identity() -> Shaping:
    """Return a shaping function that leaves raw rewards unchanged."""
    return lambda value: value


def quadratic(*, target: float, width: float = 1.0) -> Shaping:
    """Build a quadratic penalty toward a target.

    Args:
        target: The value at which the penalty is 0.
        width: Tolerance as a fraction of the target, absolute when the target is 0.

    Returns:
        See quadratic_penalty.

    Raises:
        ValueError: If width is not positive.
    """
    tolerance(target, width)  # validate now, not on the first training step
    return lambda value: quadratic_penalty(value, target, width)


def gaussian(*, target: float, width: float = 1.0) -> Shaping:
    """Build a bounded bell curve peaked at a target.

    Args:
        target: The value at which the score is 1.
        width: Tolerance as a fraction of the target, absolute when the target is 0.

    Returns:
        See gaussian_score.

    Raises:
        ValueError: If width is not positive.
    """
    tolerance(target, width)  # validate now, not on the first training step
    return lambda value: gaussian_score(value, target, width)
