"""Tests for the composite (weighted-sum) reward.

The arithmetic is checked against an explicit from-scratch expectation: every term contributes
weight * shaping(reward) and nothing else does. The cases cover each way a term can enter the
total -- weighted, shaped or raw, zero-weighted, named by an alias or by a path, with or without
arguments -- plus the validation that rejects a malformed term at build time.
"""

import math

import pytest
from omegaconf import OmegaConf

from idiom.train.grpo.reward import (
    build_reward,
    build_terms,
    entropy,
    quadratic_penalty,
)

FIXTURES = "tests.reward_fixtures"

# Two terms a run would typically name; nothing puts them in an objective by itself.
BASE = [
    {"reward": "entropy", "weight": 0.1, "shaping": {"name": "quadratic", "target": 3.68, "width": 0.2}},
    {"reward": "length", "weight": 0.1, "shaping": {"name": "quadratic", "target": 100, "width": 1.0}},
]


def _cfg(terms):
    """The configs/grpo.yaml reward structure."""
    return OmegaConf.create({"terms": terms})


def test_weighted_sum_matches_explicit_arithmetic():
    cfg = _cfg(BASE + [{"reward": f"{FIXTURES}:fraction_proline", "weight": 2.0},
                       {"reward": {"name": f"{FIXTURES}:scaled", "residue": "P", "scale": 0.005},
                        "label": "half", "weight": 3.0}])
    idr = "P" * 100  # fraction_proline = 1.0, scaled = 0.5, length sits exactly on the target
    totals, breakdown = build_reward(cfg)([idr], 1)
    expect = (0.1 * quadratic_penalty(entropy()([idr])[0], 3.68, 0.2)
              + 0.1 * quadratic_penalty(100.0, 100, 1.0)
              + 2.0 * 1.0
              + 3.0 * 0.5)
    assert math.isclose(totals[0], expect, abs_tol=1e-12)
    # the breakdown names every term, which is what the per-term W&B logging keys off
    assert set(breakdown[0]) == {"entropy", "entropy_raw", "length", "length_raw",
                                 "fraction_proline", "fraction_proline_raw", "half", "half_raw",
                                 "total"}
    assert math.isclose(breakdown[0]["total"], totals[0], abs_tol=1e-12)


def test_breakdown_separates_the_raw_reward_from_the_contribution():
    cfg = _cfg([{"reward": f"{FIXTURES}:fraction_alanine", "weight": 2.5}])
    _, breakdown = build_reward(cfg)(["AAAA"], 1)
    assert breakdown[0]["fraction_alanine_raw"] == 1.0    # in its own units
    assert breakdown[0]["fraction_alanine"] == 2.5        # weight * shaping(raw)


def test_shaping_is_applied_before_the_weight():
    cfg = _cfg([{"reward": {"name": f"{FIXTURES}:scaled", "residue": "P", "scale": 0.1},
                 "label": "frac", "weight": 1.0,
                 "shaping": {"name": "quadratic", "target": 0.15, "width": 1.0}}])
    totals, _ = build_reward(cfg)(["PPP"], 1)   # raw = 0.30
    assert math.isclose(totals[0], quadratic_penalty(0.30, 0.15, 1.0), abs_tol=1e-12)


def test_a_reward_with_no_shaping_passes_its_raw_value_through():
    cfg = _cfg([{"reward": f"{FIXTURES}:fraction_proline", "weight": 1.0}])
    totals, _ = build_reward(cfg)(["PPAA"], 1)
    assert totals == [0.5]


def test_zero_weight_term_is_logged_but_not_optimized():
    cfg = _cfg([{"reward": {"name": f"{FIXTURES}:scaled", "scale": 7.0}, "label": "watch",
                 "weight": 0.0}])
    totals, breakdown = build_reward(cfg)(["P"], 1)
    assert totals == [0.0]                       # contributes nothing to the objective
    assert breakdown[0]["watch_raw"] == 7.0      # and is still scored and logged


# --- one uniform term shape ---------------------------------------------------------------------
# A term is four keys, and its reward and shaping are named the same way: a bare name, or a mapping
# of a name plus that factory's own arguments.


def test_a_bare_name_and_a_mapping_are_the_same_term():
    bare = build_terms(_cfg([{"reward": "entropy", "weight": 1.0}]))
    mapping = build_terms(_cfg([{"reward": {"name": "entropy"}, "weight": 1.0}]))
    assert bare[0].reward(["ACDE"]) == mapping[0].reward(["ACDE"])


def test_a_reward_takes_its_settings_from_the_term():
    terms = build_terms(_cfg([{"reward": {"name": f"{FIXTURES}:scaled", "residue": "A",
                                          "scale": 2.0}, "label": "a", "weight": 1.0}]))
    assert terms[0].reward(["AAP"]) == [4.0]


def test_the_label_defaults_to_the_reward_name():
    terms = build_terms(_cfg([{"reward": "entropy", "weight": 1.0},
                              {"reward": f"{FIXTURES}:fraction_proline", "weight": 1.0},
                              {"reward": {"name": "length"}, "label": "n", "weight": 1.0}]))
    assert [t.label for t in terms] == ["entropy", "fraction_proline", "n"]


def test_shaping_defaults_to_identity():
    terms = build_terms(_cfg([{"reward": "length", "weight": 1.0}]))
    assert terms[0].shaping(3.7) == 3.7


def test_unknown_term_key_is_rejected():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['cmd'\]"):
        build_terms(_cfg([{"reward": "entropy", "cmd": "true", "weight": 1.0}]))


def test_a_reward_setting_left_at_the_term_level_is_rejected():
    # timeout belongs inside reward, next to the scorer's name
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['timeout'\]"):
        build_terms(_cfg([{"reward": {"name": "scorer", "cmd": "true"}, "label": "x",
                           "timeout": 30.0, "weight": 1.0}]))


def test_a_term_without_a_reward_is_rejected():
    with pytest.raises(ValueError, match="a term needs a reward"):
        build_terms(_cfg([{"weight": 1.0, "shaping": "identity"}]))


def test_unknown_reward_name_lists_the_shipped_ones():
    with pytest.raises(ValueError, match=r"unknown reward 'nope'.*entropy"):
        build_terms(_cfg([{"reward": "nope", "weight": 1.0}]))


def test_bad_arguments_name_the_factory():
    with pytest.raises(ValueError, match=r"bad arguments for reward .*scaled"):
        build_terms(_cfg([{"reward": {"name": f"{FIXTURES}:scaled", "nope": 1}, "weight": 1.0}]))


def test_a_reward_that_is_not_a_factory_is_rejected(tmp_path):
    # the common slip: a function that scores an IDR, rather than one that builds the scorer
    mod = tmp_path / "flat.py"
    mod.write_text("def score(idr='' ):\n    return float(len(idr))\n")
    with pytest.raises(ValueError, match="must be a factory returning a callable"):
        build_terms(_cfg([{"reward": f"{mod}:score", "label": "x", "weight": 1.0}]))


@pytest.mark.parametrize("spec, match", [
    ("no.such.module:f", "cannot import"),
    (f"{FIXTURES}:no_such_function", "has no attribute"),
    ("/no/such/file.py:f", "cannot import"),
])
def test_an_unimportable_reward_fails_at_build_time(spec, match):
    with pytest.raises(ValueError, match=match):
        build_terms(_cfg([{"reward": spec, "label": "x", "weight": 1.0}]))


def test_duplicate_labels_are_rejected():
    with pytest.raises(ValueError, match="duplicate label"):
        build_terms(_cfg([{"reward": "entropy", "weight": 1.0},
                          {"reward": "length", "label": "entropy", "weight": 1.0}]))


def test_empty_terms_is_rejected():
    # an objective with no terms gives every completion the same reward, so GRPO has no signal;
    # the library adds nothing of its own, so this is a config error rather than a silent no-op
    with pytest.raises(ValueError, match="reward.terms is empty"):
        build_reward(_cfg([]))


def test_missing_terms_key_is_rejected_like_an_empty_list():
    with pytest.raises(ValueError, match="reward.terms is empty"):
        build_terms(OmegaConf.create({}))
