"""GRPO reward functions: ``f(idr: str) -> float`` over the decoded IDR residue string.

A registry maps names → reward fns (config selects one). Composition (e.g. a base reward +
length + entropy shaping) is assembled in the GRPO module from these building blocks. ProtGPS
(localization) also fits ``f(idr) -> float`` but loads a vendored model, so it's registered
lazily in the operator path, not here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

REWARD_REGISTRY: dict[str, Callable[[str], float]] = {}


def register_reward(name: str):
    def deco(fn: Callable[[str], float]) -> Callable[[str], float]:
        REWARD_REGISTRY[name] = fn
        return fn

    return deco


def get_reward(name: str) -> Callable[[str], float]:
    if name not in REWARD_REGISTRY:
        raise KeyError(f"unknown reward {name!r}; registered: {sorted(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name]


def _fraction(idr: str, aa: str) -> float:
    return idr.count(aa) / len(idr) if idr else 0.0


@register_reward("fraction_proline")
def fraction_proline(idr: str) -> float:
    return _fraction(idr, "P")


@register_reward("fraction_alanine")
def fraction_alanine(idr: str) -> float:
    return _fraction(idr, "A")


def sequence_entropy(idr: str) -> float:
    """Shannon entropy (**bits**) of the IDR's amino-acid composition (max log2(20) ≈ 4.32 bits)."""
    if not idr:
        return 0.0
    n = len(idr)
    return -sum((c / n) * math.log2(c / n) for c in Counter(idr).values())


def length_reward(idr: str, *, target_length: int, width: float = 1.0) -> float:
    """Quadratic penalty (legacy): ``-((len - target)/(target*width))^2``, max 0 at target."""
    if not idr:
        return -1.0  # max penalty for an empty IDR (legacy)
    d = (len(idr) - target_length) / (target_length * width)
    return -(d * d)


def entropy_reward(idr: str, *, target_entropy: float = 3.68, width: float = 1.0) -> float:
    """Quadratic penalty (legacy): ``-((H - target)/(target*width))^2``, max 0 at target.

    ``H`` and ``target_entropy`` are in **bits** (see :func:`sequence_entropy`). The default 3.68
    bits = 2.55 nats (the corpus-matched target; AFDB IDR mean ~3.64 bits). Because the penalty is a
    ratio, the nats->bits switch leaves the reward (and training dynamics) unchanged as long as the
    configured target is converted too.
    """
    d = (sequence_entropy(idr) - target_entropy) / (target_entropy * width)  # H=0 for empty IDR
    return -(d * d)


def quadratic_shaping(raw: float, *, target: float, scale: float = 1.0) -> float:
    """Reward shaping toward a target raw value: ``1 - scale*(raw - target)^2``."""
    return 1.0 - scale * (raw - target) ** 2
