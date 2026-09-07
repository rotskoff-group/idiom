"""Tests for the shaping rules that turn a raw reward into a shaped one.

Each rule is checked against the arithmetic it promises, and the spec resolver against the
malformed specs it must reject at build time.
"""

import math

import pytest
from omegaconf import OmegaConf

from idiom.train.grpo.reward import (
    SHAPING_ALIASES,
    build_from_spec,
    build_reward,
    gaussian_score,
    quadratic_penalty,
    tolerance,
)


def build_shaping(spec):
    """Resolve a term's shaping spec the way compose does."""
    return build_from_spec(spec or "identity", SHAPING_ALIASES, "shaping", "shaping")


def test_tolerance_is_relative_to_the_target_and_absolute_at_zero():
    assert tolerance(25.0, 0.2) == 5.0
    assert tolerance(0.0, 0.5) == 0.5         # nothing to be relative to: width is absolute
    assert tolerance(-4.0, 0.5) == 2.0
    with pytest.raises(ValueError, match="width must be positive"):
        tolerance(25.0, 0.0)


def test_quadratic_penalty_is_zero_at_the_target_and_unbounded_away():
    assert quadratic_penalty(25.0, 25.0, 0.2) == 0.0
    assert quadratic_penalty(30.0, 25.0, 0.2) == pytest.approx(-1.0)   # one tolerance out
    assert quadratic_penalty(20.0, 25.0, 0.2) == pytest.approx(-1.0)
    assert quadratic_penalty(100.0, 25.0, 0.2) == pytest.approx(-225.0)  # does not saturate


def test_gaussian_score_is_bounded():
    assert gaussian_score(25.0, 25.0, 0.2) == pytest.approx(1.0)
    assert gaussian_score(30.0, 25.0, 0.2) == pytest.approx(math.exp(-1.0))
    assert gaussian_score(1e6, 25.0, 0.2) == 0.0


def test_build_shaping_applies_a_spec():
    shaping = build_shaping({"name": "quadratic", "target": 100, "width": 1.0})
    assert [shaping(v) for v in (100.0, 200.0)] == [0.0, pytest.approx(-1.0)]


def test_build_shaping_without_a_spec_is_identity():
    identity = build_shaping(None)
    assert [identity(v) for v in (0.25, 3.0)] == [0.25, 3.0]


def test_build_shaping_rejects_a_bad_spec():
    with pytest.raises(ValueError, match=r"unknown shaping 'quadratik'.*gaussian"):
        build_shaping({"name": "quadratik", "target": 1})
    with pytest.raises(ValueError, match="needs a name"):
        build_shaping({"target": 1})
    with pytest.raises(ValueError, match="bad arguments"):
        build_shaping({"name": "quadratic"})               # quadratic without a target
    with pytest.raises(ValueError, match="bad arguments"):
        build_shaping({"name": "quadratic", "target": 1, "min": 2})  # an argument it does not take
    with pytest.raises(ValueError, match="width must be positive"):
        build_shaping({"name": "quadratic", "target": 1, "width": 0})


def test_a_term_can_name_a_shaping_rule_of_its_own(tmp_path):
    mod = tmp_path / "my_shaping.py"
    mod.write_text(
        "from idiom.train.grpo.reward import tolerance\n"
        "def one_sided(*, target, width=1.0):\n"
        "    scale = tolerance(target, width)\n"
        "    return lambda v: -(((target - v) / scale) ** 2) if v < target else 0.0\n"
    )
    cfg = OmegaConf.create({"terms": [
        {"reward": "length", "weight": 1.0,
         "shaping": {"name": f"{mod}:one_sided", "target": 10, "width": 0.5}},
    ]})
    totals, _ = build_reward(cfg)(["A" * 20, "A" * 10, "A" * 5], 1)
    assert totals == [0.0, 0.0, pytest.approx(-1.0)]  # flat above the threshold, penalized below
