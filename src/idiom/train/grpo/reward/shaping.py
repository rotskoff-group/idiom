"""Reward shaping: how a raw reward becomes a shaped one.

A shaping rule is a factory returning f(raw) -> float, named in a term exactly as a reward is:

    shaping: {name: quadratic, target: 100, width: 1.0}
    shaping: identity                                    # or omit shaping entirely

quadratic is 0 at the target and unbounded below; gaussian is 1 at the target and bounded in
[0, 1]; identity passes the raw reward through. All three say *be here*, so a threshold or a band
is a rule of your own -- write a factory of the same shape and name it by its "module:function"
path, as in cookbook/rewards/custom_rewards.py.
"""

from __future__ import annotations

import math
from collections.abc import Callable

Shaping = Callable[[float], float]


def tolerance(target: float, width: float) -> float:
    """Return the deviation that counts as one width away from a target.

    The tolerance is relative to the target, so width=0.2 means 20%; when the target is 0, width is
    used as an absolute tolerance.

    Args:
        target (float): The value being aimed at.
        width (float): Tolerance as a fraction of the target, or an absolute tolerance when the
            target is 0.

    Returns:
        float: The absolute deviation corresponding to one width.

    Raises:
        ValueError: If width is not positive.
    """
    if width <= 0:
        raise ValueError(f"shaping width must be positive, got {width!r}")
    return abs(target) * width if target else width


def quadratic_penalty(value: float, target: float, width: float = 1.0) -> float:
    """Score a value against a target with an unbounded quadratic penalty.

    Args:
        value (float): The raw reward.
        target (float): The value at which the penalty is 0.
        width (float): Tolerance as a fraction of the target.

    Returns:
        float: 0 at the target, -1 one tolerance away, decreasing without bound beyond that.
    """
    return -((value - target) / tolerance(target, width)) ** 2


def gaussian_score(value: float, target: float, width: float = 1.0) -> float:
    """Score a value against a target with a bounded bell curve.

    Args:
        value (float): The raw reward.
        target (float): The value at which the score is 1.
        width (float): Tolerance as a fraction of the target.

    Returns:
        float: 1 at the target, about 0.37 one tolerance away, and flattening to 0 far from it.
    """
    return math.exp(-((value - target) / tolerance(target, width)) ** 2)


def identity() -> Shaping:
    """Build the shaping rule that passes the raw reward through unchanged.

    Returns:
        Shaping: The identity function.
    """
    return lambda value: value


def quadratic(*, target: float, width: float = 1.0) -> Shaping:
    """Build a quadratic penalty toward a target.

    Args:
        target (float): The value at which the penalty is 0.
        width (float): Tolerance as a fraction of the target, absolute when the target is 0.

    Returns:
        Shaping: See quadratic_penalty.

    Raises:
        ValueError: If width is not positive.
    """
    tolerance(target, width)  # validate now, not on the first training step
    return lambda value: quadratic_penalty(value, target, width)


def gaussian(*, target: float, width: float = 1.0) -> Shaping:
    """Build a bounded bell curve peaked at a target.

    Args:
        target (float): The value at which the score is 1.
        width (float): Tolerance as a fraction of the target, absolute when the target is 0.

    Returns:
        Shaping: See gaussian_score.

    Raises:
        ValueError: If width is not positive.
    """
    tolerance(target, width)  # validate now, not on the first training step
    return lambda value: gaussian_score(value, target, width)
