"""Tests for the composite (weighted-sum) reward.

Two things are checked: the new term-block config sums exactly as specified, and every legacy config
shape (name / group / shaping / monitor + length + entropy) reproduces bit-for-bit through the
desugaring, so old configs and old checkpoints train identically. The reference is a from-scratch
reimplementation of the pre-refactor arithmetic, kept in this file.
"""

import math

from omegaconf import OmegaConf

from idiom.train.grpo.rewards import (
    entropy_reward, get_reward, length_reward, quadratic_shaping, register_group_reward,
    register_reward)
from idiom.train.grpo.train_grpo import build_reward, build_reward_terms


# ---- reference: the exact pre-refactor per-idr composition ----------------------------------


def _legacy_total(rcfg, idr):
    base = get_reward(rcfg["name"])(idr)
    total = base
    if rcfg["shaping"]["enabled"]:
        total = quadratic_shaping(total, target=rcfg["shaping"]["target"], scale=rcfg["shaping"]["scale"])
    if rcfg["length"]["enabled"]:
        total += rcfg["length"]["weight"] * length_reward(
            idr, target_length=rcfg["length"]["target_length"], width=rcfg["length"]["width"])
    if rcfg["entropy"]["enabled"]:
        total += rcfg["entropy"]["weight"] * entropy_reward(
            idr, target_entropy=rcfg["entropy"]["target_entropy"], width=rcfg["entropy"]["width"])
    return total


def _legacy_cfg(**over):
    base = {
        "module": None,
        "name": "fraction_proline",
        "group": None,
        "monitor": None,
        "shaping": {"enabled": False, "target": 0.9, "scale": 1.0},
        "length": {"enabled": True, "target_length": 100, "width": 1.0, "weight": 2.0},
        "entropy": {"enabled": True, "target_entropy": 3.68, "width": 0.2, "weight": 0.5},
    }
    base.update(over)
    return base


IDRS = ["", "P" * 100, "PPPAAAKKK", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GSGS" * 30]


# ---- legacy configs reproduce exactly -------------------------------------------------------


def test_legacy_default_matches_reference():
    cfg = _legacy_cfg()
    reward = build_reward(OmegaConf.create(cfg))
    for idr in IDRS:
        assert reward(idr) == _legacy_total(cfg, idr)


def test_legacy_with_shaping_matches_reference():
    cfg = _legacy_cfg(shaping={"enabled": True, "target": 0.5, "scale": 2.0})
    reward = build_reward(OmegaConf.create(cfg))
    for idr in IDRS:
        assert math.isclose(reward(idr), _legacy_total(cfg, idr), rel_tol=0, abs_tol=1e-12)


def test_legacy_grid_matches_reference():
    for shaping in (False, True):
        for length in (False, True):
            for entropy in (False, True):
                cfg = _legacy_cfg(
                    shaping={"enabled": shaping, "target": 0.9, "scale": 1.0},
                    length={"enabled": length, "target_length": 80, "width": 0.5, "weight": 1.5},
                    entropy={"enabled": entropy, "target_entropy": 3.5, "width": 0.3, "weight": 0.7},
                )
                reward = build_reward(OmegaConf.create(cfg))
                for idr in IDRS:
                    assert math.isclose(reward(idr), _legacy_total(cfg, idr), abs_tol=1e-12), (
                        shaping, length, entropy, idr)


def test_legacy_group_reward_matches_reference():
    # a legacy group reward: base scored on the whole batch, then per-idr length/entropy added,
    # shaping never applied (the old group path forced it off)
    register_group_reward("_grp_len")(lambda idrs, gs: [float(len(x)) for x in idrs])
    cfg = _legacy_cfg(name=None, group="_grp_len")
    terms = build_reward_terms(OmegaConf.create(cfg))
    totals, _ = terms(IDRS, 1)
    for idr, total in zip(IDRS, totals):
        expect = float(len(idr))
        expect += cfg["length"]["weight"] * length_reward(idr, target_length=100, width=1.0)
        expect += cfg["entropy"]["weight"] * entropy_reward(idr, target_entropy=3.68, width=0.2)
        assert math.isclose(total, expect, abs_tol=1e-12)


def test_legacy_monitor_is_logged_not_optimized():
    register_reward("_mon")(lambda idr: 42.0)
    cfg = _legacy_cfg(monitor="_mon")
    terms = build_reward_terms(OmegaConf.create(cfg))
    totals, breakdown = terms(["PPPP"], 1)
    assert breakdown[0]["_mon"] == 42.0            # logged
    assert math.isclose(totals[0], _legacy_total(cfg, "PPPP"), abs_tol=1e-12)  # not in total


# ---- new term-block config ------------------------------------------------------------------


def _new_cfg(**over):
    base = {
        "module": None,
        "entropy": {"enabled": True, "weight": 0.1, "target_entropy": 3.68, "width": 0.2},
        "length": {"enabled": True, "weight": 0.1, "target_length": 100, "width": 1.0},
        "rl_sae": {"enabled": False, "weight": 1.0, "signature": "nucleolus"},
        "external": [],
    }
    base.update(over)
    return OmegaConf.create(base)


def test_new_weighted_sum():
    register_reward("_r_prol")(lambda idr: idr.count("P") / len(idr) if idr else 0.0)
    register_reward("_r_half")(lambda idr: 0.5)
    cfg = _new_cfg(
        external=[{"enabled": True, "weight": 2.0, "name": "_r_prol"},
                  {"enabled": True, "weight": 3.0, "name": "_r_half"}],
    )
    terms = build_reward_terms(cfg)
    idr = "P" * 100  # _r_prol = 1.0, length penalty 0 at target, entropy fixed
    totals, breakdown = terms([idr], 1)
    expect = (0.1 * entropy_reward(idr, target_entropy=3.68, width=0.2)
              + 0.1 * length_reward(idr, target_length=100, width=1.0)
              + 2.0 * 1.0
              + 3.0 * 0.5)
    assert math.isclose(totals[0], expect, abs_tol=1e-12)
    assert set(breakdown[0]) == {"entropy", "length", "_r_prol", "_r_half", "total"}


def test_rl_sae_term_resolves_signature(tmp_path):
    # the rl_sae block imports its module (default rewards/rl_sae.py) and resolves sae_only_<sig>
    mod = tmp_path / "fake_rl_sae.py"
    mod.write_text(
        "from idiom.train.grpo.rewards import register_reward\n"
        "register_reward('sae_only_myco')(lambda idr: 0.7 if idr else 0.0)\n"
    )
    cfg = _new_cfg(
        entropy={"enabled": False, "weight": 0.0, "target_entropy": 3.68, "width": 0.2},
        length={"enabled": False, "weight": 0.0, "target_length": 100, "width": 1.0},
        rl_sae={"enabled": True, "weight": 2.0, "signature": "myco", "module": str(mod)},
    )
    totals, breakdown = build_reward_terms(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 2.0 * 0.7, abs_tol=1e-12)
    assert "rl_sae" in breakdown[0]


def test_new_multiple_external_rewards():
    register_reward("_r1")(lambda idr: 1.0)
    register_reward("_r2")(lambda idr: 10.0)
    cfg = _new_cfg(
        entropy={"enabled": False, "weight": 0.1, "target_entropy": 3.68, "width": 0.2},
        length={"enabled": False, "weight": 0.1, "target_length": 100, "width": 1.0},
        external=[{"enabled": True, "weight": 1.0, "name": "_r1"},
                  {"enabled": True, "weight": 0.5, "name": "_r2"}],
    )
    totals, _ = build_reward_terms(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 1.0 * 1.0 + 0.5 * 10.0, abs_tol=1e-12)


def test_new_disabled_terms_drop_out():
    cfg = _new_cfg(
        entropy={"enabled": False, "weight": 0.1, "target_entropy": 3.68, "width": 0.2},
        length={"enabled": False, "weight": 0.1, "target_length": 100, "width": 1.0},
    )
    totals, breakdown = build_reward_terms(cfg)(["ACDE"], 1)
    assert totals == [0.0] and breakdown[0] == {"total": 0.0}  # nothing enabled -> zero reward


def test_new_external_batched_reward():
    # a batched (group-registered) external reward is called once for the whole batch
    calls = []
    register_group_reward("_batched")(lambda idrs, gs: calls.append(len(idrs)) or [float(len(x)) for x in idrs])
    cfg = _new_cfg(
        entropy={"enabled": False, "weight": 0.0, "target_entropy": 3.68, "width": 0.2},
        length={"enabled": False, "weight": 0.0, "target_length": 100, "width": 1.0},
        external=[{"enabled": True, "weight": 1.0, "name": "_batched"}],
    )
    totals, _ = build_reward_terms(cfg)(["AA", "CCC", "DDDD"], 3)
    assert totals == [2.0, 3.0, 4.0]
    assert calls == [3]  # one call for the whole batch, not one per completion


def test_new_monitor_term():
    register_reward("_watch")(lambda idr: 7.0)
    cfg = _new_cfg(
        entropy={"enabled": False, "weight": 0.0, "target_entropy": 3.68, "width": 0.2},
        length={"enabled": False, "weight": 0.0, "target_length": 100, "width": 1.0},
        external=[{"enabled": True, "weight": 99.0, "name": "_watch", "monitor": True}],
    )
    totals, breakdown = build_reward_terms(cfg)(["AA"], 1)
    assert totals == [0.0]                  # monitor excluded from total despite weight
    assert breakdown[0]["_watch"] == 7.0    # raw score logged
