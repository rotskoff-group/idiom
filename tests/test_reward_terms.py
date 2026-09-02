"""Tests for the composite (weighted-sum) reward.

The reward is the whole GRPO objective, so the arithmetic is checked against an explicit
from-scratch expectation rather than against itself: every enabled term contributes weight * score
and nothing else does. The cases below cover each way a term can enter or leave the total —
enabled/disabled, weighted, monitor-only, in-process vs. batched — because a term that silently
drops out (or is double-counted) changes what the policy optimizes without failing anywhere.
"""

import math

from omegaconf import OmegaConf

from idiom.train.grpo.reward import entropy_reward, length_reward, register_reward
from idiom.train.grpo.train_grpo import build_reward_terms


def _cfg(**over):
    """The configs/grpo.yaml term-block shape, with the guardrails on by default."""
    base = {
        "module": None,
        "entropy": {"enabled": True, "weight": 0.1, "target_entropy": 3.68, "width": 0.2},
        "length": {"enabled": True, "weight": 0.1, "target_length": 100, "width": 1.0},
        "rl_sae": {"enabled": False, "weight": 1.0, "signature": "nucleolus"},
        "external": [],
    }
    base.update(over)
    return OmegaConf.create(base)


OFF = {"entropy": {"enabled": False, "weight": 0.0, "target_entropy": 3.68, "width": 0.2},
       "length": {"enabled": False, "weight": 0.0, "target_length": 100, "width": 1.0}}


def test_weighted_sum_matches_explicit_arithmetic():
    register_reward("_r_prol")(lambda idr: idr.count("P") / len(idr) if idr else 0.0)
    register_reward("_r_half")(lambda idr: 0.5)
    cfg = _cfg(external=[{"enabled": True, "weight": 2.0, "name": "_r_prol"},
                         {"enabled": True, "weight": 3.0, "name": "_r_half"}])
    idr = "P" * 100  # _r_prol = 1.0, and length sits exactly on the target
    totals, breakdown = build_reward_terms(cfg)([idr], 1)
    expect = (0.1 * entropy_reward(idr, target_entropy=3.68, width=0.2)
              + 0.1 * length_reward(idr, target_length=100, width=1.0)
              + 2.0 * 1.0
              + 3.0 * 0.5)
    assert math.isclose(totals[0], expect, abs_tol=1e-12)
    # the breakdown names every enabled term, which is what the per-term W&B logging keys off
    assert set(breakdown[0]) == {"entropy", "length", "_r_prol", "_r_half", "total"}
    assert math.isclose(breakdown[0]["total"], totals[0], abs_tol=1e-12)


def test_breakdown_entries_are_weighted_contributions():
    register_reward("_r_one")(lambda idr: 1.0)
    cfg = _cfg(**OFF, external=[{"enabled": True, "weight": 2.5, "name": "_r_one"}])
    _, breakdown = build_reward_terms(cfg)(["ACDE"], 1)
    assert breakdown[0]["_r_one"] == 2.5  # weight * score, not the raw score


def test_multiple_external_rewards_sum():
    register_reward("_r1")(lambda idr: 1.0)
    register_reward("_r2")(lambda idr: 10.0)
    cfg = _cfg(**OFF, external=[{"enabled": True, "weight": 1.0, "name": "_r1"},
                                {"enabled": True, "weight": 0.5, "name": "_r2"}])
    totals, _ = build_reward_terms(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 1.0 * 1.0 + 0.5 * 10.0, abs_tol=1e-12)


def test_disabled_terms_drop_out_entirely():
    register_reward("_r_off")(lambda idr: 99.0)
    cfg = _cfg(**OFF, external=[{"enabled": False, "weight": 1.0, "name": "_r_off"}])
    totals, breakdown = build_reward_terms(cfg)(["ACDE"], 1)
    assert totals == [0.0] and breakdown[0] == {"total": 0.0}


def test_monitor_term_is_logged_but_not_optimized():
    register_reward("_watch")(lambda idr: 7.0)
    cfg = _cfg(**OFF, external=[{"enabled": True, "weight": 99.0, "name": "_watch", "monitor": True}])
    totals, breakdown = build_reward_terms(cfg)(["AA"], 1)
    assert totals == [0.0]                  # excluded from the total despite the weight
    assert breakdown[0]["_watch"] == 7.0    # and logged raw, not weighted


def test_duplicate_term_labels_stay_distinct():
    # two entries can name the same reward; their log keys must not collide and overwrite each other
    register_reward("_dup")(lambda idr: 1.0)
    cfg = _cfg(**OFF, external=[{"enabled": True, "weight": 1.0, "name": "_dup"},
                                {"enabled": True, "weight": 2.0, "name": "_dup"}])
    totals, breakdown = build_reward_terms(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 3.0, abs_tol=1e-12)
    assert len(breakdown[0]) == 3  # two distinct term keys + total


def test_module_is_imported_before_lookup(tmp_path):
    # a user drops a *.py with @register_reward; reward.module imports it so the name resolves
    mod = tmp_path / "myrew.py"
    mod.write_text(
        "from idiom.train.grpo.reward import register_reward\n"
        "register_reward('_from_module')(lambda idr: 4.0)\n"
    )
    cfg = _cfg(**OFF, module=str(mod),
               external=[{"enabled": True, "weight": 0.5, "name": "_from_module"}])
    totals, _ = build_reward_terms(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 2.0, abs_tol=1e-12)


def test_rl_sae_term_resolves_signature(tmp_path):
    # the rl_sae block resolves sae_only_<signature>; module overrides the bundled reward so the
    # test never loads a real SAE
    mod = tmp_path / "fake_rl_sae.py"
    mod.write_text(
        "from idiom.train.grpo.reward import register_reward\n"
        "register_reward('sae_only_myco')(lambda idr: 0.7 if idr else 0.0)\n"
    )
    cfg = _cfg(**OFF,
               rl_sae={"enabled": True, "weight": 2.0, "signature": "myco", "module": str(mod)})
    totals, breakdown = build_reward_terms(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 2.0 * 0.7, abs_tol=1e-12)
    assert "rl_sae" in breakdown[0]


def test_scores_stay_aligned_across_a_batch():
    # totals[i] must correspond to idrs[i]; a term that reorders would corrupt the advantages
    register_reward("_len")(lambda idr: float(len(idr)))
    cfg = _cfg(**OFF, external=[{"enabled": True, "weight": 1.0, "name": "_len"}])
    idrs = ["A", "AA", "AAA", ""]
    totals, breakdown = build_reward_terms(cfg)(idrs, 2)
    assert totals == [1.0, 2.0, 3.0, 0.0]
    assert [b["total"] for b in breakdown] == totals
