"""Composition of the total GRPO reward from its configured terms.

build_reward turns the reward config's terms list into a single function mapping (idrs, group_size)
to per-idr totals and a matching per-term breakdown. Every term follows one rule,

    total reward = sum over terms of weight * shaping(reward(idrs))

The library adds nothing of its own: reward.terms is empty by default and a run names every term it
optimizes, so the launch line is the whole objective. A term with weight 0 keeps running and stays
logged but has no influence.

The config is validated here, before the model is loaded.
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

# A term names its raw reward in one of two ways, and each source has keys the other has no use
# for. Naming a key from the wrong group is an error rather than a silent no-op.
SHARED_KEYS = {"reward", "cmd", "label", "weight", "shaping"}
NAMED_KEYS = {"module", "params", "batched"}      # reward: a registry name or "module:function"
EXTERNAL_KEYS = {"timeout", "maxlen", "cwd", "env"}  # cmd: a scorer subprocess
TERM_KEYS = SHARED_KEYS | NAMED_KEYS | EXTERNAL_KEYS


@dataclass(frozen=True)
class RewardTermSpec:
    """One validated reward term.

    Attributes:
        label (str): Name the term is logged under; unique among the terms.
        weight (float): Multiplier on the shaped reward; 0 logs the term without optimizing it.
        reward (str | None): Registry key of the raw reward, or None for a cmd term.
        cmd (str | list[str] | None): External scorer command, or None for a registered term.
        shaping (dict | None): Shaping spec, or None for identity.
        params (dict | None): Keyword arguments for a "module:function" factory, which is called
            with them to produce the reward itself; None calls the function directly on each idr.
        batched (bool): For a "module:function" reward, True if the callable already takes
            (idrs, batch) and returns one value per idr, rather than one idr at a time.
        timeout (float): Seconds to wait for one external response.
        maxlen (int): Truncate sequences to this length before sending them to a scorer; 0 sends
            them whole.
        cwd (str | None): Working directory for an external scorer.
        env (dict | None): Environment variables set for an external scorer, over this process's
            own environment.
    """

    label: str
    weight: float
    reward: str | None = None
    cmd: str | list[str] | None = None
    shaping: dict | None = None
    params: dict | None = None
    batched: bool = False
    timeout: float = 300.0
    maxlen: int = 0
    cwd: str | None = None
    env: dict | None = None


def import_module_spec(spec: str | None):
    """Import a user module so its register_reward decorators run.

    Args:
        spec (str | None): A dotted module path or a path ending in ".py"; None or empty is a
            no-op.

    Returns:
        ModuleType | None: The imported module, or None if spec was empty.
    """
    if not spec:
        return None
    if spec.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location("idiom_custom_rewards", spec)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def is_callable_spec(reward: str | None) -> bool:
    """Return whether a term's reward names a callable directly rather than a registry key.

    Args:
        reward (str | None): The term's reward field.

    Returns:
        bool: True for the "module:function" form, which carries a ":".
    """
    return bool(reward) and ":" in reward


def load_callable(reward: str):
    """Import and return the callable a "module:function" reward names.

    The module part is either a dotted name or a path to a .py file, as in import_module_spec.

    Args:
        reward (str): A reward of the form "package.module:function" or "path/to/file.py:function".

    Returns:
        tuple[str, Callable]: The attribute name, and the callable itself.

    Raises:
        ValueError: If the module or the attribute cannot be imported, or the attribute is not
            callable.
    """
    mod_name, _, attr = reward.rpartition(":")
    if not mod_name or not attr:
        raise ValueError(f"reward {reward!r} is not of the form 'module:function'")
    try:
        module = import_module_spec(mod_name)
    except Exception as e:  # ImportError, FileNotFoundError, or anything the module raises
        raise ValueError(f"reward {reward!r}: cannot import {mod_name!r} ({e})") from e
    try:
        fn = getattr(module, attr)
    except AttributeError:
        raise ValueError(f"reward {reward!r}: {mod_name!r} has no attribute {attr!r}") from None
    if not callable(fn):
        raise ValueError(f"reward {reward!r}: {attr!r} is not callable (got {type(fn).__name__})")
    return attr, fn


def _parse_term(raw: dict, where: str) -> RewardTermSpec:
    """Validate one term and return its spec.

    Any module the term names is imported here, before its reward name is looked up, and the reward
    itself is resolved -- so a bad path or an unregistered name fails now rather than on the first
    training step.

    Args:
        raw (dict): One entry of reward.terms.
        where (str): The term's config path, used in error messages.

    Returns:
        RewardTermSpec: The validated term.

    Raises:
        ValueError: If the term carries an unknown key, carries a key belonging to the other kind
            of source, names neither or both of reward and cmd, gives a cmd no label, names an
            unregistered reward, or names a "module:function" callable that cannot be imported.
    """
    term = dict(raw)
    unknown = set(term) - TERM_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; "
                         f"a term takes {sorted(TERM_KEYS)}")

    reward, cmd = term.pop("reward", None), term.pop("cmd", None)
    if (reward is None) == (cmd is None):
        raise ValueError(f"{where}: give exactly one of reward (a registered name) or cmd "
                         f"(an external scorer command)")
    # Every term has a label, a weight and a shaping; the rest belongs to one source or the other,
    # so a key from the wrong group is reported rather than quietly ignored.
    misplaced = sorted(set(term) & (NAMED_KEYS if cmd is not None else EXTERNAL_KEYS))
    if misplaced:
        has, belongs = ("cmd scorer", "a reward") if cmd is not None else ("reward", "a cmd")
        raise ValueError(f"{where}: key(s) {misplaced} do not apply to a {has} term; they belong "
                         f"to {belongs} term")

    import_module_spec(term.pop("module", None))  # before the reward name is looked up

    # A "module:function" reward is logged under the function's own name, since the whole dotted
    # spec makes an unreadable metric key; rpartition leaves a plain registry name untouched.
    label = term.pop("label", None) or (reward.rpartition(":")[2] if reward else None)
    if not label:
        raise ValueError(f"{where}: a cmd term needs a label to log it under")

    factory = is_callable_spec(reward)  # "module:function", the only form params and batched fit
    params = term.pop("params", None)
    if params is not None and not factory:
        raise ValueError(f"{where}: params applies only to a 'module:function' reward, which is "
                         f"then called with them as a factory; a registered reward takes its "
                         f"settings at register_reward time")
    if term.get("batched") and not factory:
        raise ValueError(f"{where}: batched applies only to a 'module:function' reward; a "
                         f"registered reward declares it at register_reward(batched=True), and a "
                         f"cmd scorer is always sent the whole batch")
    if factory:
        load_callable(reward)   # import now, so a bad path fails here, not mid-run
    elif reward is not None:
        get_reward(reward)      # fail here, by name, rather than on the first training step

    return RewardTermSpec(label=label, weight=float(term.pop("weight", 1.0)),
                          reward=reward, cmd=cmd, params=params, **term)


def _check_unique_labels(specs: list[RewardTermSpec]) -> None:
    """Reject two terms logged under one label, which would collide in the breakdown and in W&B.

    Args:
        specs (list[RewardTermSpec]): The parsed terms, in config order.

    Raises:
        ValueError: If two terms share a label.
    """
    seen: dict[str, int] = {}
    for i, spec in enumerate(specs):
        if spec.label in seen:
            raise ValueError(f"reward.terms[{i}]: duplicate label {spec.label!r}, already used by "
                             f"reward.terms[{seen[spec.label]}]; give one term its own label")
        seen[spec.label] = i


def parse_terms(rcfg: DictConfig) -> list[RewardTermSpec]:
    """Validate a reward config and return its terms.

    The config-level module is imported first, so a reward or shaping name it registers resolves
    for every term; each term is then validated on its own by _parse_term.

    Args:
        rcfg (DictConfig): The reward config: an optional module and a terms list.

    Returns:
        list[RewardTermSpec]: One validated spec per term, in config order.

    Raises:
        ValueError: If the terms list is empty, two terms share a label, or a term does not
            validate; see _parse_term.
    """
    cfg = OmegaConf.to_container(rcfg, resolve=True) if isinstance(rcfg, DictConfig) else dict(rcfg)
    import_module_spec(cfg.get("module"))

    terms = list(cfg.get("terms") or [])
    if not terms:
        raise ValueError(
            "reward.terms is empty; a run names every term it optimizes. Pass the whole objective "
            "at launch, e.g. reward.terms='[{reward: entropy, weight: 1.0, shaping: {type: "
            "quadratic, target: 3.65, width: 0.2}}]'; see cookbook/scripts/grpo/ for a ready-to-"
            "submit script per objective.")

    specs = [_parse_term(t, f"reward.terms[{i}]") for i, t in enumerate(terms)]
    _check_unique_labels(specs)
    return specs


def _source(spec: RewardTermSpec) -> Callable[[list[str], Batch], list[float]]:
    """Return the batched function producing a term's raw rewards."""
    if is_callable_spec(spec.reward):
        fn = load_callable(spec.reward)[1]
        if spec.params is not None:
            fn = fn(**spec.params)  # a factory: the config's params build the reward itself
        if spec.batched:
            return lambda idrs, batch: [float(v) for v in fn(idrs, batch)]
        return lambda idrs, batch: [float(fn(idr)) for idr in idrs]
    if spec.reward is not None:
        return get_reward(spec.reward)
    return make_external_reward(spec.cmd, timeout=spec.timeout, maxlen=spec.maxlen,
                                 cwd=spec.cwd, env=spec.env, label=spec.label)


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
        ValueError: If the config does not validate; see parse_terms and build_shaping.
    """
    terms = [(s.label, s.weight, _source(s), build_shaping(s.shaping)) for s in parse_terms(rcfg)]

    def score_batch(idrs: list[str], group_size: int):
        batch = Batch(group_size=group_size)
        totals = [0.0] * len(idrs)
        breakdown: list[dict[str, float]] = [{} for _ in idrs]
        for label, weight, source, shaping in terms:
            for i, raw_i in enumerate(source(idrs, batch)):
                shaped_i = weight * shaping(raw_i)
                breakdown[i][f"{label}_raw"] = raw_i     # the raw reward, in its own units
                breakdown[i][label] = shaped_i           # its contribution to the objective
                totals[i] += shaped_i
        for i, total in enumerate(totals):
            breakdown[i]["total"] = total
        return totals, breakdown

    return score_batch
