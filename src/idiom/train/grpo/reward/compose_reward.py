"""Composition of the GRPO reward from its configured terms.

build_reward_terms turns a reward config — the entropy, length, rl_sae, and external blocks — into
a single function mapping (idrs, group_size) to per-idr totals and a matching per-term breakdown.
Every term is scored a whole batch at a time; per-idr rewards are looped behind that signature,
while batched terms such as an external scorer are called once per step.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable

from omegaconf import DictConfig

from idiom.train.grpo.reward.base import (
    entropy_reward, length_reward, resolve_reward)
from idiom.train.grpo.reward.external_reward import make_external_reward


def _register_custom_rewards(spec: str | None) -> None:
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


def build_reward_terms(rcfg: DictConfig):
    """Build the composite reward from a reward config.

    The total for each completion is the sum of weight * score over the enabled terms:

        total = w_entropy * entropy + w_length * length + w_rl_sae * rl_sae + sum(w_i * external_i)

    Each external entry is either a registered in-process reward, named by "name", or a subprocess
    scorer given by "cmd". A term with monitor set to True is included in the breakdown but
    excluded from the total.

    Args:
        rcfg (DictConfig): Reward config: an optional module plus the entropy, length, rl_sae, and
            external blocks (see configs/grpo.yaml).

    Returns:
        Callable[[list[str], int], tuple[list[float], list[dict[str, float]]]]: Maps
            (idrs, group_size) to the per-idr totals and a matching per-term breakdown (each dict
            holds every enabled term keyed by its log name, plus total).
    """
    _register_custom_rewards(rcfg.get("module"))  # import user rewards before lookup

    # (log_key, weight, scorer(idrs, group_size) -> list[float], monitor)
    terms: list[tuple[str, float, Callable, bool]] = []

    ent = rcfg.get("entropy")
    if ent and ent.get("enabled"):
        te, wd = ent.target_entropy, ent.width
        terms.append(("entropy", float(ent.weight),
                      lambda idrs, gs, te=te, wd=wd:
                          [entropy_reward(x, target_entropy=te, width=wd) for x in idrs], False))

    ln = rcfg.get("length")
    if ln and ln.get("enabled"):
        tl, wd = ln.target_length, ln.width
        terms.append(("length", float(ln.weight),
                      lambda idrs, gs, tl=tl, wd=wd:
                          [length_reward(x, target_length=tl, width=wd) for x in idrs], False))

    rs = rcfg.get("rl_sae")
    if rs and rs.get("enabled"):
        if rs.get("module"):
            _register_custom_rewards(rs.get("module"))  # user override of the bundled reward
        else:
            import idiom.train.grpo.reward.rl_sae_reward  # registers sae_only_<sig> on import
        terms.append(("rl_sae", float(rs.weight), resolve_reward(f"sae_only_{rs.signature}"),
                      bool(rs.get("monitor"))))

    # external rewards ("bring your own"): each entry is EITHER a subprocess scorer (cmd, run in its
    # own environment) OR a registered in-process reward (name, e.g. a custom reward from
    # rewards/example_rewards.py). A per-entry module is imported first so its name resolves.
    seen = {k for k, *_ in terms}
    for i, ext in enumerate(rcfg.get("external") or []):
        if not ext.get("enabled"):
            continue
        if ext.get("module"):
            _register_custom_rewards(ext.get("module"))
        label = ext.get("name") or f"external{i}"
        key = label if label not in seen else f"{label}_{i}"
        seen.add(key)
        if ext.get("cmd"):
            scorer = make_external_reward(
                ext.cmd, target=ext.get("target"), width=float(ext.get("width", 1.0)),
                timeout=float(ext.get("timeout", 300.0)), maxlen=int(ext.get("maxlen", 0)), label=key)
        else:
            scorer = resolve_reward(ext.name)
        terms.append((key, float(ext.weight), scorer, bool(ext.get("monitor"))))

    def score_batch(idrs: list[str], group_size: int):
        n = len(idrs)
        totals = [0.0] * n
        breakdown: list[dict[str, float]] = [{} for _ in range(n)]
        for key, weight, scorer, monitor in terms:
            scores = scorer(idrs, group_size)
            for i in range(n):
                if monitor:
                    breakdown[i][key] = scores[i]          # logged, excluded from total
                else:
                    val = weight * scores[i]
                    breakdown[i][key] = val
                    totals[i] += val
        for i in range(n):
            breakdown[i]["total"] = totals[i]
        return totals, breakdown

    return score_batch
