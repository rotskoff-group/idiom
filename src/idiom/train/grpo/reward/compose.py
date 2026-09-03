"""Composition of the total GRPO reward from its configured terms.

build_reward turns the reward config's terms list into a single function mapping (idrs, group_size)
to per-idr totals and a matching per-term breakdown. Every term follows one rule,

    total reward = sum over terms of weight * shaping(reward(idrs))

so there is nothing special about a built-in term, a user's registered reward, or an external scorer
subprocess: they differ only in where the raw reward comes from.

Two ways to take a term out of the objective, and they are not the same one. weight 0 keeps the term
running and logged, which is how a diagnostic is watched without letting it steer training. enabled
false skips it entirely -- nothing imported, no subprocess, no per-step cost -- which is what lets
the shipped config carry a menu of ready-to-use terms that cost nothing until switched on.

The config is validated here, before the model is loaded, so a typo fails in seconds rather than
after a queue wait.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from idiom.train.grpo.reward.external import make_external_reward
from idiom.train.grpo.reward.registry import Batch, get_reward
from idiom.train.grpo.reward.shaping import build_shaping

TERM_KEYS = {"enabled", "reward", "cmd", "label", "weight", "shaping", "module", "timeout",
             "maxlen", "cwd"}


@dataclass(frozen=True)
class TermSpec:
    """One validated reward term.

    Attributes:
        label (str): Name the term is logged under; unique among the enabled terms.
        weight (float): Multiplier on the shaped reward; 0 logs the term without optimizing it.
        reward (str | None): Registry key of the raw reward, or None for a cmd term.
        cmd (str | list[str] | None): External scorer command, or None for a registered term.
        shaping (dict | None): Shaping spec, or None for identity.
        timeout (float): Seconds to wait for one external response.
        maxlen (int): Truncate sequences to this length before sending them to a scorer; 0 sends
            them whole.
        cwd (str | None): Working directory for an external scorer.
    """

    label: str
    weight: float
    reward: str | None = None
    cmd: str | list[str] | None = None
    shaping: dict | None = None
    timeout: float = 300.0
    maxlen: int = 0
    cwd: str | None = None


def import_module_spec(spec: str | None) -> None:
    """Import a user module so its register_reward decorators run.

    Args:
        spec (str | None): A dotted module path or a path ending in ".py"; None or empty is a
            no-op.
    """
    if not spec:
        return
    if spec.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location("idiom_custom_rewards", spec)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        importlib.import_module(spec)


def parse_terms(rcfg: DictConfig) -> list[TermSpec]:
    """Validate a reward config and return its terms.

    Any module named by the config, or by a term, is imported first so that reward names resolve.

    A term with enabled false is dropped before anything else happens to it, so a disabled term
    costs nothing and is not validated: it may name a reward this environment cannot import, which
    is what makes a config full of switched-off examples usable.

    Args:
        rcfg (DictConfig): The reward config: an optional module plus a terms list.

    Returns:
        list[TermSpec]: One validated spec per term, in config order.

    Raises:
        ValueError: If an enabled term carries an unknown key, names neither or both of reward and
            cmd, gives a cmd no label, repeats a label, or names an unregistered reward.
    """
    cfg = OmegaConf.to_container(rcfg, resolve=True) if isinstance(rcfg, DictConfig) else dict(rcfg)
    import_module_spec(cfg.get("module"))

    specs: list[TermSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(cfg.get("terms") or []):
        where = f"reward.terms[{i}]"
        term = dict(raw)
        if not term.pop("enabled", True):
            continue  # skipped whole: nothing imported, no subprocess, nothing validated
        unknown = set(term) - TERM_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; "
                             f"a term takes {sorted(TERM_KEYS)}")
        import_module_spec(term.pop("module", None))  # before the reward name is looked up

        reward, cmd = term.pop("reward", None), term.pop("cmd", None)
        if (reward is None) == (cmd is None):
            raise ValueError(f"{where}: give exactly one of reward (a registered name) or cmd "
                             f"(an external scorer command)")
        label = term.pop("label", None) or reward
        if not label:
            raise ValueError(f"{where}: a cmd term needs a label to log it under")
        if label in seen:
            raise ValueError(f"{where}: duplicate label {label!r}; give one term its own label")
        seen.add(label)
        if reward is not None:
            get_reward(reward)  # fail here, by name, rather than on the first training step

        specs.append(TermSpec(label=label, weight=float(term.pop("weight", 1.0)),
                              reward=reward, cmd=cmd, **term))
    return specs


def _source(spec: TermSpec) -> Callable[[list[str], Batch], list[float]]:
    """Return the batched function producing a term's raw rewards."""
    if spec.reward is not None:
        return get_reward(spec.reward)
    return make_external_reward(spec.cmd, timeout=spec.timeout, maxlen=spec.maxlen,
                                 cwd=spec.cwd, label=spec.label)


def build_reward(rcfg: DictConfig):
    """Build the total reward from a reward config.

    Args:
        rcfg (DictConfig): The reward config: an optional module plus a terms list (see
            configs/grpo.yaml).

    Returns:
        Callable[[list[str], int], tuple[list[float], list[dict[str, float]]]]: Maps
            (idrs, group_size) to the per-idr totals and a matching breakdown. Each breakdown dict
            holds every term's weighted contribution under its label, its raw reward under
            "<label>_raw", and the total.

    Raises:
        ValueError: If the config does not validate (see parse_terms and build_shaping).
    """
    terms = [(s.label, s.weight, _source(s), build_shaping(s.shaping)) for s in parse_terms(rcfg)]

    def score_batch(idrs: list[str], group_size: int):
        batch = Batch(group_size=group_size)
        totals = [0.0] * len(idrs)
        breakdown: list[dict[str, float]] = [{} for _ in idrs]
        for label, weight, source, shaping in terms:
            raw = source(idrs, batch)
            shaped = shaping(raw, batch)
            for i, (raw_i, shaped_i) in enumerate(zip(raw, shaped)):
                breakdown[i][f"{label}_raw"] = raw_i         # the raw reward, in its own units
                breakdown[i][label] = weight * shaped_i      # its contribution to the objective
                totals[i] += weight * shaped_i
        for i, total in enumerate(totals):
            breakdown[i]["total"] = total
        return totals, breakdown

    return score_batch
