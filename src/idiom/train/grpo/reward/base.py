"""GRPO reward functions of the form f(idr: str) -> float over the decoded IDR residue string.

A registry maps names to reward functions; the composite reward (a weighted sum of terms) is
assembled in reward/compose_reward.py from these building blocks. Rewards that need their own environment
run as external subprocess scorers instead (reward/external_reward.py).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

REWARD_REGISTRY: dict[str, Callable[[str], float]] = {}


def register_reward(name: str):
    """Return a decorator that registers a reward function under name.

    Args:
        name (str): Registry key for the decorated reward function.

    Returns:
        Callable: A decorator that registers an f(idr: str) -> float and returns it unchanged.
    """

    def deco(fn: Callable[[str], float]) -> Callable[[str], float]:
        REWARD_REGISTRY[name] = fn
        return fn

    return deco


def get_reward(name: str) -> Callable[[str], float]:
    """Look up a registered reward function by name.

    Args:
        name (str): Registry key of the reward function.

    Returns:
        Callable[[str], float]: The registered reward function.

    Raises:
        KeyError: If no reward is registered under name.
    """
    if name not in REWARD_REGISTRY:
        raise KeyError(f"unknown reward {name!r}; registered: {sorted(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name]


def resolve_reward(name: str) -> Callable[[list[str], int], list[float]]:
    """Return a batch scorer f(idrs, group_size) -> list[float] for a registered per-idr reward.

    The per-idr reward is lifted to the batch signature by looping. This uniform signature is how the
    composite reward sums per-idr terms (entropy, length, rl_sae) and batched terms (external
    scorers, which score the whole step in one call) in one place.

    Args:
        name (str): Registry key of a per-idr reward.

    Returns:
        Callable[[list[str], int], list[float]]: One score per IDR, in order.

    Raises:
        KeyError: If no reward is registered under name.
    """
    if name in REWARD_REGISTRY:
        fn = REWARD_REGISTRY[name]
        return lambda idrs, group_size: [fn(idr) for idr in idrs]
    raise KeyError(f"unknown reward {name!r}; registered: {sorted(REWARD_REGISTRY)}")


def sequence_entropy(idr: str) -> float:
    """Return the Shannon entropy in bits of the IDR's amino-acid composition.

    The maximum is log2(20), about 4.32 bits.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Composition entropy in bits (0.0 for an empty IDR).
    """
    if not idr:
        return 0.0
    n = len(idr)
    return -sum((c / n) * math.log2(c / n) for c in Counter(idr).values())


def quadratic_penalty(value: float, target: float, width: float) -> float:
    """Quadratic penalty -((value - target) / (target * width))^2: 0 at the target, negative away.

    The shared shaping for every target-seeking term (length, entropy, and external scorers). width
    is a tolerance relative to the target, so the penalty reaches -1 at a deviation of width*|target|
    and its magnitude is scale-free (choosing bits vs nats, or A vs nm, leaves it unchanged as long
    as the target is converted too). When target is 0 the relative scale is undefined, so width is
    used as an absolute tolerance instead.

    Args:
        value (float): The measured value to score.
        target (float): The value at which the penalty is 0.
        width (float): Tolerance as a fraction of the target (absolute when target is 0).

    Returns:
        float: 0 at the target, increasingly negative (unbounded) as value moves away.
    """
    scale = target * width if target else width  # relative tolerance; absolute when target == 0
    d = (value - target) / scale
    return -(d * d)


def length_reward(idr: str, *, target_length: int, width: float = 1.0) -> float:
    """Quadratic length penalty -((len - target_length) / (target_length * width))^2, max 0 at target.

    Args:
        idr (str): The decoded IDR residue string.
        target_length (int): Desired IDR length.
        width (float): Tolerance as a fraction of the target length.

    Returns:
        float: The penalty (0 at the target length, -1.0 for an empty IDR).
    """
    if not idr:
        return -1.0  # an empty IDR has no length to score; -1.0 is the one-tolerance-out penalty
    return quadratic_penalty(len(idr), target_length, width)


def entropy_reward(idr: str, *, target_entropy: float = 3.65, width: float = 1.0) -> float:
    """Quadratic entropy penalty -((H - target_entropy) / (target_entropy * width))^2, max 0 at target.

    H and target_entropy are in bits (see sequence_entropy). The default 3.65 bits equals about 2.53
    nats (the corpus-matched target; AFDB IDR mean about 3.64 bits). Because the penalty is a
    ratio, switching from nats to bits leaves the reward (and training dynamics) unchanged as long
    as the configured target is converted too.

    Args:
        idr (str): The decoded IDR residue string.
        target_entropy (float): Desired composition entropy in bits.
        width (float): Tolerance as a fraction of the target entropy.

    Returns:
        float: The penalty (0 at the target entropy).
    """
    return quadratic_penalty(sequence_entropy(idr), target_entropy, width)  # H=0 for empty IDR

