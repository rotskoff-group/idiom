"""The reward registry and the built-in reward terms.

A reward is a function f(idr: str) -> float over a decoded IDR residue string. register_reward
adds one to REWARD_REGISTRY under a name, and get_reward and resolve_reward look it up.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

REWARD_REGISTRY: dict[str, Callable[[str], float]] = {}


def register_reward(name: str):
    """Return a decorator that registers a reward function under name.

    An existing entry with the same name is replaced.

    Args:
        name (str): Registry key for the decorated reward function.

    Returns:
        Callable: A decorator that registers f(idr: str) -> float and returns it unchanged.
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
    """Look up a per-idr reward and lift it to the batch scorer signature.

    Args:
        name (str): Registry key of a per-idr reward.

    Returns:
        Callable[[list[str], int], list[float]]: A function mapping (idrs, group_size) to one
            score per IDR, in order.

    Raises:
        KeyError: If no reward is registered under name.
    """
    if name in REWARD_REGISTRY:
        fn = REWARD_REGISTRY[name]
        return lambda idrs, group_size: [fn(idr) for idr in idrs]
    raise KeyError(f"unknown reward {name!r}; registered: {sorted(REWARD_REGISTRY)}")


def sequence_entropy(idr: str) -> float:
    """Return the Shannon entropy of an IDR's amino-acid composition, in bits.

    The value ranges from 0 for a single repeated residue to log2(20), about 4.32 bits, for a
    uniform composition.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Composition entropy in bits, or 0.0 for an empty string.
    """
    if not idr:
        return 0.0
    n = len(idr)
    return -sum((c / n) * math.log2(c / n) for c in Counter(idr).values())


def quadratic_penalty(value: float, target: float, width: float) -> float:
    """Score a value against a target with an unbounded quadratic penalty.

    The penalty is 0 at the target and reaches -1 at a deviation of width * |target|. When target
    is 0, width is used as an absolute tolerance instead of a relative one.

    Args:
        value (float): The measured value to score.
        target (float): The value at which the penalty is 0.
        width (float): Tolerance as a fraction of the target, or an absolute tolerance when the
            target is 0.

    Returns:
        float: 0 at the target, decreasing quadratically away from it.
    """
    scale = target * width if target else width  # relative tolerance; absolute when target == 0
    d = (value - target) / scale
    return -(d * d)


def length_reward(idr: str, *, target_length: int, width: float = 1.0) -> float:
    """Score an IDR's length against a target with a quadratic penalty.

    Args:
        idr (str): The decoded IDR residue string.
        target_length (int): Desired IDR length.
        width (float): Tolerance as a fraction of the target length.

    Returns:
        float: 0 at the target length, decreasing away from it, and -1.0 for an empty string.
    """
    if not idr:
        return -1.0  # an empty IDR has no length to score; -1.0 is the one-tolerance-out penalty
    return quadratic_penalty(len(idr), target_length, width)


def entropy_reward(idr: str, *, target_entropy: float = 3.65, width: float = 1.0) -> float:
    """Score an IDR's composition entropy against a target with a quadratic penalty.

    Both the measured entropy and target_entropy are in bits. The default target of 3.65 bits is
    close to the mean composition entropy of the training corpus.

    Args:
        idr (str): The decoded IDR residue string.
        target_entropy (float): Desired composition entropy in bits.
        width (float): Tolerance as a fraction of the target entropy.

    Returns:
        float: 0 at the target entropy, decreasing away from it. An empty string scores as
            entropy 0.
    """
    return quadratic_penalty(sequence_entropy(idr), target_entropy, width)  # H=0 for empty IDR

