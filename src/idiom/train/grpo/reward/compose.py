"""Composition of the total GRPO reward from its configured terms.

build_reward turns the reward config's terms list into a single function mapping (idrs, group_size)
to per-idr totals and a matching per-term breakdown. Every term follows one rule,

    total reward = sum over terms of weight * shaping(reward(idrs))

There are two ways to take a term out of the objective: weight 0 keeps it running and logged but
gives it no influence, while enabled false skips it entirely -- nothing imported, no subprocess.

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

TERM_KEYS = {"enabled", "reward", "cmd", "label", "weight", "shaping", "module", "timeout",
             "maxlen", "cwd", "batched"}


@dataclass(frozen=True)
class RewardTermSpec:
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
        batched (bool): For a "module:function" reward, True if the callable already takes
            (idrs, batch) and returns one value per idr, rather than one idr at a time.
    """

    label: str
    weight: float
    reward: str | None = None
    cmd: str | list[str] | None = None
    shaping: dict | None = None
    timeout: float = 300.0
    maxlen: int = 0
    cwd: str | None = None
    batched: bool = False


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


def resolve_add(cfg: dict) -> list[dict]:
    """Expand the reward config's `add` selector into terms, appended after `terms`.

    An entry is either a name from the `presets` menu or a whole term written out, so a run picks
    what it optimizes at launch: `reward.add=[rg]` selects a preset and
    `reward.presets.rg.shaping.target=30` tunes it.

    Args:
        cfg (dict): The reward config, holding an optional `add` list and `presets` mapping.

    Returns:
        list[dict]: The selected terms, in the order they were named.

    Raises:
        ValueError: If a name is not in the menu, or an entry is neither a name nor a mapping.
    """
    presets = cfg.get("presets") or {}
    out: list[dict] = []
    for i, entry in enumerate(cfg.get("add") or []):
        if isinstance(entry, dict):
            out.append(entry)  # a whole term, written out at the command line
            continue
        if not isinstance(entry, str):
            raise ValueError(f"reward.add[{i}]: expected a preset name or a term mapping, got "
                             f"{type(entry).__name__}")
        if entry not in presets:
            raise ValueError(f"reward.add[{i}]: unknown preset {entry!r}; the menu is "
                             f"{sorted(presets)}. Pass a whole term instead to use your own.")
        out.append(dict(presets[entry]))
    return out


def parse_terms(rcfg: DictConfig) -> list[RewardTermSpec]:
    """Validate a reward config and return its terms.

    Any module named by the config, or by a term, is imported first so reward names resolve. A term
    with enabled false is dropped before validation, so it may name a reward this environment
    cannot import.

    Args:
        rcfg (DictConfig): The reward config: an optional module, a terms list, and an optional
            `add` selector naming presets to append (see resolve_add).

    Returns:
        list[RewardTermSpec]: One validated spec per term, in config order.

    Raises:
        ValueError: If an enabled term carries an unknown key, names neither or both of reward and
            cmd, gives a cmd no label, repeats a label, names an unregistered reward, or names a
            "module:function" callable that cannot be imported.
    """
    cfg = OmegaConf.to_container(rcfg, resolve=True) if isinstance(rcfg, DictConfig) else dict(rcfg)
    import_module_spec(cfg.get("module"))

    # The guardrails in `terms`, then whatever this run asked for by name in `add`.
    terms = list(cfg.get("terms") or []) + resolve_add(cfg)

    specs: list[RewardTermSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(terms):
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
        # A "module:function" reward is logged under the function's own name, since the whole
        # dotted spec makes an unreadable metric key.
        default_label = reward.rpartition(":")[2] if is_callable_spec(reward) else reward
        label = term.pop("label", None) or default_label
        if not label:
            raise ValueError(f"{where}: a cmd term needs a label to log it under")
        if label in seen:
            raise ValueError(f"{where}: duplicate label {label!r}; give one term its own label")
        seen.add(label)
        if is_callable_spec(reward):
            load_callable(reward)  # import now, so a bad path fails here, not mid-run
        elif reward is not None:
            get_reward(reward)  # fail here, by name, rather than on the first training step
        if term.get("batched") and not is_callable_spec(reward):
            raise ValueError(f"{where}: batched applies only to a 'module:function' reward; a "
                             f"registered reward declares it at register_reward(batched=True), and "
                             f"a cmd scorer is always sent the whole batch")

        specs.append(RewardTermSpec(label=label, weight=float(term.pop("weight", 1.0)),
                                    reward=reward, cmd=cmd, **term))
    return specs


def _source(spec: RewardTermSpec) -> Callable[[list[str], Batch], list[float]]:
    """Return the batched function producing a term's raw rewards."""
    if is_callable_spec(spec.reward):
        fn = load_callable(spec.reward)[1]
        if spec.batched:
            return lambda idrs, batch: [float(v) for v in fn(idrs, batch)]
        return lambda idrs, batch: [float(fn(idr)) for idr in idrs]
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
