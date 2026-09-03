"""The reward registry.

A reward is a function of a decoded IDR residue string returning one raw value, in whatever unit
suits it -- bits, residues, angstroms, a fraction. Deciding what a good value is happens separately
(see shaping) and is chosen by config, so one reward serves as a target, a guardrail, or a
logged-only diagnostic without being rewritten.

Three qualified names run through this package, because the bare word is ambiguous once a term is
assembled: the *raw reward* is what a registered function returns, the *shaped reward* is that value
after shaping, and the *total reward* is the weighted sum over terms that GRPO optimizes.

register_reward accepts either form:

    @register_reward("aromatic_fraction")                 # f(idr) -> float, lifted to a batch
    @register_reward("sae_only_nucleolus", batched=True)  # f(idrs, batch) -> list[float]

The batched form is for rewards that cost less per batch than per sequence, such as a GPU forward
pass or a subprocess round trip.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

REWARD_REGISTRY: dict[str, Callable[[list[str], "Batch"], list[float]]] = {}


@dataclass(frozen=True)
class Batch:
    """What a reward or its shaping knows about the batch being scored.

    Attributes:
        group_size (int): Number of completions sampled per prompt. Consecutive runs of this many
            IDRs form one GRPO group, which is what a batched reward needs to score a completion
            against the others it was sampled with.
    """

    group_size: int = 1


def register_reward(name: str, *, batched: bool = False):
    """Return a decorator that registers a reward under name.

    An existing entry with the same name is replaced.

    Args:
        name (str): Registry key for the decorated function.
        batched (bool): True if the function already takes (idrs, batch) and returns one raw value
            per IDR; False for a per-idr f(idr) -> float, which is lifted over the batch.

    Returns:
        Callable: A decorator that registers the function and returns it unchanged.
    """

    def deco(fn):
        REWARD_REGISTRY[name] = fn if batched else _lift(fn)
        return fn

    return deco


def _lift(fn: Callable[[str], float]) -> Callable[[list[str], Batch], list[float]]:
    """Lift a per-idr reward to the batch signature."""
    return lambda idrs, batch: [float(fn(idr)) for idr in idrs]


def get_reward(name: str) -> Callable[[list[str], Batch], list[float]]:
    """Look up a registered reward by name.

    Args:
        name (str): Registry key of the reward.

    Returns:
        Callable[[list[str], Batch], list[float]]: The batched reward.

    Raises:
        KeyError: If no reward is registered under name.
    """
    if name not in REWARD_REGISTRY:
        raise KeyError(f"unknown reward {name!r}; registered: {sorted(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name]
