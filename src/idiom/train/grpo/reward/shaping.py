"""Reward shaping: how a raw reward becomes a shaped one.

A shaping rule is chosen per term in the config, so the same reward can be pushed toward a target or
passed through untouched:

    shaping: {type: quadratic, target: 100, width: 1.0}
    (omitted)                                          # identity

quadratic is unbounded below, so a term far from its target can swamp the rest of the sum; gaussian
is the bounded alternative when several targets have to coexist, at the cost of flattening far from
the target, where it stops distinguishing bad from worse. zscore normalizes within each GRPO group
and is the way to combine rewards whose scales you do not know in advance.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import fmean, pstdev

from idiom.train.grpo.reward.registry import Batch

# type -> factory(**params) -> shaping(values, batch) -> list[float]
SHAPING_REGISTRY: dict[str, Callable[..., Callable[[list[float], Batch], list[float]]]] = {}


def register_shaping(shaping_type: str, *, elementwise: bool = True):
    """Return a decorator that registers a shaping factory under type.

    Args:
        shaping_type (str): The name used in a term's shaping.type.
        elementwise (bool): True if the factory returns a scalar f(value) -> float, which is lifted
            over the batch; False if it returns f(values, batch) -> list[float] directly, which a
            group-relative shaping rule needs.

    Returns:
        Callable: A decorator that registers the factory and returns it unchanged.
    """

    def deco(factory):
        if elementwise:
            def batched_factory(**params):
                f = factory(**params)
                return lambda values, batch: [f(v) for v in values]
            SHAPING_REGISTRY[shaping_type] = batched_factory
        else:
            SHAPING_REGISTRY[shaping_type] = factory
        return factory

    return deco


def tolerance(target: float, width: float) -> float:
    """Return the deviation that counts as one width away from a target.

    The tolerance is relative to the target, so width=0.2 means 20%. When the target is 0 there is
    nothing to be relative to, and width is used as an absolute tolerance instead.

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


@register_shaping("identity")
def _identity():
    """Build the shaping rule that passes the raw reward through unchanged."""
    return lambda value: value


@register_shaping("quadratic")
def _quadratic(*, target: float, width: float = 1.0):
    """Build a quadratic penalty toward target."""
    return lambda value: quadratic_penalty(value, target, width)


@register_shaping("gaussian")
def _gaussian(*, target: float, width: float = 1.0):
    """Build a bounded bell curve peaked at target."""
    return lambda value: gaussian_score(value, target, width)


@register_shaping("zscore", elementwise=False)
def _zscore():
    """Build the shaping rule that standardizes values within each GRPO group.

    The batch is split into consecutive groups of batch.group_size, matching the rollout layout;
    a group whose values are all equal scores 0, since no completion in it is better than another.
    """

    def shaping(values: list[float], batch: Batch) -> list[float]:
        g = batch.group_size
        if g < 2 or len(values) % g:  # not a clean grouping (e.g. a unit test); use the whole batch
            g = len(values)
        out: list[float] = []
        for start in range(0, len(values), g):
            chunk = values[start:start + g]
            sd = pstdev(chunk) if len(chunk) > 1 else 0.0
            mean = fmean(chunk)
            out += [0.0] * len(chunk) if sd == 0 else [(v - mean) / sd for v in chunk]
        return out

    return shaping


def build_shaping(spec) -> Callable[[list[float], Batch], list[float]]:
    """Build a shaping rule from a term's shaping spec.

    Args:
        spec: A mapping with a "type" plus that shaping type's parameters, or None for identity.

    Returns:
        Callable[[list[float], Batch], list[float]]: Maps raw rewards to shaped ones.

    Raises:
        ValueError: If the shaping type is missing or unknown, or its parameters do not fit it.
    """
    if not spec:
        return SHAPING_REGISTRY["identity"]()
    params = dict(spec)
    shaping_type = params.pop("type", None)
    if shaping_type is None:
        raise ValueError(f"shaping needs a type (one of {sorted(SHAPING_REGISTRY)})")
    if shaping_type not in SHAPING_REGISTRY:
        raise ValueError(
            f"unknown shaping type {shaping_type!r}; known types: {sorted(SHAPING_REGISTRY)}"
        )
    try:
        return SHAPING_REGISTRY[shaping_type](**params)
    except TypeError as e:  # a missing target, or a parameter this type does not take
        raise ValueError(f"bad parameters for shaping {shaping_type!r}: {e}") from e
