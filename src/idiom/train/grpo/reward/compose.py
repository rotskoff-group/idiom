"""Compose rewards as sum(weight * shaping(raw_reward)).

Every term is explicit; zero-weight terms are still evaluated and logged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from idiom.train.grpo.reward.resolve import (
    REWARD_ALIASES,
    SHAPING_ALIASES,
    Reward,
    build_from_spec,
    spec_name,
)
from idiom.train.grpo.reward.shaping import Shaping

TERM_KEYS = {"reward", "shaping", "weight", "label"}


def _finite(value, where: str) -> float:
    """Convert a score or weight to a finite float, identifying invalid values."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{where}: expected a finite number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{where}: expected a finite number, got {value!r}")
    return number


@dataclass(frozen=True)
class Term:
    """A weighted reward and its shaping function.

    Attributes:
        label: Name the term is logged under; unique among the terms.
        weight: Multiplier on the shaped reward; 0 logs the term without optimizing it.
        reward: Maps a step's IDRs to one raw value each.
        shaping: Maps one raw value to its shaped value.
    """

    label: str
    weight: float
    reward: Reward
    shaping: Shaping


def _parse_term(raw: dict, where: str) -> tuple[str, float, object, object]:
    """Validate a term and return its label, weight, reward spec, and shaping spec."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: a term is a mapping of {sorted(TERM_KEYS)}, got "
                         f"{type(raw).__name__}")
    term = dict(raw)
    unknown = set(term) - TERM_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; a term takes "
                         f"{sorted(TERM_KEYS)}. A reward's own settings go inside reward, next to "
                         f"its name.")

    reward_spec = term.get("reward")
    if reward_spec is None:
        raise ValueError(f"{where}: a term needs a reward -- a shipped name, a 'module:function' "
                         f"path, or a mapping of either plus that factory's arguments")
    # Use the factory name as the default metric label.
    label = term.get("label") or spec_name(reward_spec, f"{where}.reward")[0].rpartition(":")[2]
    weight = _finite(term.get("weight", 1.0), f"{where} ({label!r}).weight")
    return label, weight, reward_spec, term.get("shaping")


def _check_unique_labels(labels: list[str]) -> None:
    """Reject duplicate metric labels with ValueError."""
    seen: dict[str, int] = {}
    for i, label in enumerate(labels):
        if label in seen:
            raise ValueError(f"reward.terms[{i}]: duplicate label {label!r}, already used by "
                             f"reward.terms[{seen[label]}]; give one term its own label")
        seen[label] = i


def build_terms(rcfg: DictConfig) -> list[Term]:
    """Validate all term specs and labels before importing their factories.

    Args:
        rcfg: The reward config: a terms list.

    Returns:
        One built term per config entry, in order.

    Raises:
        ValueError: If the terms list is empty, two terms share a label, or a term does not
            validate; see _parse_term and resolve.build_from_spec.
    """
    cfg = OmegaConf.to_container(rcfg, resolve=True) if isinstance(rcfg, DictConfig) else dict(rcfg)
    raw_terms = list(cfg.get("terms") or [])
    if not raw_terms:
        raise ValueError(
            "reward.terms is empty; a run names every term it optimizes. Pass the whole objective "
            "at launch, e.g. reward.terms='[{reward: entropy, weight: 1.0, shaping: {name: "
            "quadratic, target: 3.65, width: 0.2}}]'; see cookbook/scripts/grpo/ for a ready-to-"
            "submit script per objective.")

    parsed = [_parse_term(t, f"reward.terms[{i}]") for i, t in enumerate(raw_terms)]
    _check_unique_labels([label for label, _, _, _ in parsed])

    terms = []
    for i, (label, weight, reward_spec, shaping_spec) in enumerate(parsed):
        where = f"reward.terms[{i}]"
        terms.append(Term(
            label=label,
            weight=weight,
            reward=build_from_spec(reward_spec, REWARD_ALIASES, "reward", where),
            shaping=build_from_spec(shaping_spec or "identity", SHAPING_ALIASES, "shaping", where),
        ))
    return terms


def build_reward(rcfg: DictConfig):
    """Build the total reward from a reward config.

    Args:
        rcfg: The reward config: a terms list (see configs/grpo.yaml).

    Returns:
        Callable[[list[str], int], tuple[list[float], list[dict[str, float]]]]: Maps (idrs,
            group_size) to the per-idr totals and a matching breakdown. Each breakdown dict holds
            every term's weighted contribution under its label, its raw reward under "<label>_raw",
            and the total.

    Raises:
        ValueError: If the config does not validate; see build_terms.
    """
    terms = build_terms(rcfg)

    def score_batch(idrs: list[str], group_size: int):
        totals = [0.0] * len(idrs)
        breakdown: list[dict[str, float]] = [{} for _ in idrs]
        for term in terms:
            values = term.reward(idrs)
            try:
                values = list(values)
            except TypeError:
                raise ValueError(f"reward {term.label!r}: expected one score per sequence") from None
            if len(values) != len(idrs):
                raise ValueError(f"reward {term.label!r}: returned {len(values)} scores for "
                                 f"{len(idrs)} sequences")
            for i, value in enumerate(values):
                where = f"reward {term.label!r}, sequence {i}"
                raw_i = _finite(value, f"{where}, raw score")
                shaped = _finite(term.shaping(raw_i), f"{where}, shaped score")
                shaped_i = _finite(term.weight * shaped, f"{where}, weighted score")
                breakdown[i][f"{term.label}_raw"] = raw_i
                breakdown[i][term.label] = shaped_i
                totals[i] = _finite(totals[i] + shaped_i, f"{where}, accumulated total")
        for i, total in enumerate(totals):
            breakdown[i]["total"] = total
        return totals, breakdown

    return score_batch
