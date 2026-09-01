"""GRPO reward functions of the form f(idr: str) -> float over the decoded IDR residue string.

A registry maps names to reward functions (the config selects one). Composition (for example a
base reward plus length and entropy shaping) is assembled in the GRPO module from these building
blocks. ProtGPS (localization) also fits f(idr) -> float but loads a vendored model, so it is
registered lazily in the operator path, not here.
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


# GROUP rewards: f(idrs: list[str], group_size: int) -> list[float]. Unlike per-idr rewards, these see
# the whole batch of completions and can score a completion RELATIVE to its GRPO group (e.g. reward
# population coverage / diversity of the SAE code rather than per-sequence cramming).
GROUP_REWARD_REGISTRY: dict[str, Callable[[list, int], list]] = {}


def register_group_reward(name: str):
    """Return a decorator that registers a group reward function under name.

    Args:
        name (str): Registry key for the decorated group reward function.

    Returns:
        Callable: A decorator that registers an f(idrs: list[str], group_size: int) -> list and
            returns it unchanged.
    """

    def deco(fn: Callable[[list, int], list]) -> Callable[[list, int], list]:
        GROUP_REWARD_REGISTRY[name] = fn
        return fn

    return deco


def get_group_reward(name: str) -> Callable[[list, int], list]:
    """Look up a registered group reward function by name.

    Args:
        name (str): Registry key of the group reward function.

    Returns:
        Callable[[list, int], list]: The registered group reward function.

    Raises:
        KeyError: If no group reward is registered under name.
    """
    if name not in GROUP_REWARD_REGISTRY:
        raise KeyError(f"unknown group reward {name!r}; registered: {sorted(GROUP_REWARD_REGISTRY)}")
    return GROUP_REWARD_REGISTRY[name]


def _fraction(idr: str, aa: str) -> float:
    return idr.count(aa) / len(idr) if idr else 0.0


@register_reward("fraction_proline")
def fraction_proline(idr: str) -> float:
    """Fraction of residues in the IDR that are proline."""
    return _fraction(idr, "P")


@register_reward("fraction_alanine")
def fraction_alanine(idr: str) -> float:
    """Fraction of residues in the IDR that are alanine."""
    return _fraction(idr, "A")


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


def length_reward(idr: str, *, target_length: int, width: float = 1.0) -> float:
    """Quadratic length penalty -((len - target_length) / (target_length * width))^2, max 0 at target.

    Args:
        idr (str): The decoded IDR residue string.
        target_length (int): Desired IDR length.
        width (float): Scale of the tolerance band around the target.

    Returns:
        float: The penalty (0 at the target length, -1.0 for an empty IDR).
    """
    if not idr:
        return -1.0  # max penalty for an empty IDR (legacy)
    d = (len(idr) - target_length) / (target_length * width)
    return -(d * d)


def entropy_reward(idr: str, *, target_entropy: float = 3.68, width: float = 1.0) -> float:
    """Quadratic entropy penalty -((H - target_entropy) / (target_entropy * width))^2, max 0 at target.

    H and target_entropy are in bits (see sequence_entropy). The default 3.68 bits equals 2.55
    nats (the corpus-matched target; AFDB IDR mean about 3.64 bits). Because the penalty is a
    ratio, switching from nats to bits leaves the reward (and training dynamics) unchanged as long
    as the configured target is converted too.

    Args:
        idr (str): The decoded IDR residue string.
        target_entropy (float): Desired composition entropy in bits.
        width (float): Scale of the tolerance band around the target.

    Returns:
        float: The penalty (0 at the target entropy).
    """
    d = (sequence_entropy(idr) - target_entropy) / (target_entropy * width)  # H=0 for empty IDR
    return -(d * d)


def quadratic_shaping(raw: float, *, target: float, scale: float = 1.0) -> float:
    """Shape a raw reward toward a target value as 1 - scale * (raw - target)^2.

    Args:
        raw (float): The base reward value to shape.
        target (float): Raw value at which the shaped reward peaks.
        scale (float): Curvature of the quadratic falloff.

    Returns:
        float: The shaped reward.
    """
    return 1.0 - scale * (raw - target) ** 2
