"""Tests for the shaping rules that turn a raw reward into a shaped one.

A shaping rule encodes the objective's intent -- hit this value or take the number as it comes --
so each type is checked against the arithmetic it promises rather than against itself.
build_shaping is checked for rejecting a malformed spec at build time, because the
whole point of validating there is that a typo must not survive until the first training step.
"""

import math

import pytest

from idiom.train.grpo.reward import (
    Batch,
    build_shaping,
    gaussian_score,
    quadratic_penalty,
    tolerance,
)


def test_tolerance_is_relative_to_the_target_and_absolute_at_zero():
    assert tolerance(25.0, 0.2) == 5.0        # 20% of 25
    assert tolerance(0.0, 0.5) == 0.5         # nothing to be relative to: width is absolute
    assert tolerance(-4.0, 0.5) == 2.0        # a negative target still gives a positive scale
    with pytest.raises(ValueError, match="width must be positive"):
        tolerance(25.0, 0.0)                  # a zero width would divide by zero


def test_quadratic_penalty_is_zero_at_the_target_and_unbounded_away():
    assert quadratic_penalty(25.0, 25.0, 0.2) == 0.0
    assert quadratic_penalty(30.0, 25.0, 0.2) == pytest.approx(-1.0)   # one tolerance out
    assert quadratic_penalty(20.0, 25.0, 0.2) == pytest.approx(-1.0)   # symmetric
    assert quadratic_penalty(100.0, 25.0, 0.2) == pytest.approx(-225.0)  # does not saturate


def test_gaussian_score_is_bounded():
    assert gaussian_score(25.0, 25.0, 0.2) == pytest.approx(1.0)
    assert gaussian_score(30.0, 25.0, 0.2) == pytest.approx(math.exp(-1.0))
    # the bounded alternative to quadratic: a term far off target cannot swamp the rest of the sum
    # -- but it flattens to 0 out there, so it carries no signal back toward the target either
    assert gaussian_score(1e6, 25.0, 0.2) == 0.0


def test_build_shaping_applies_a_spec_over_a_batch():
    shaping = build_shaping({"type": "quadratic", "target": 100, "width": 1.0})
    assert shaping([100.0, 200.0], Batch()) == [0.0, pytest.approx(-1.0)]


def test_build_shaping_without_a_spec_is_identity():
    # a reward already on a sensible scale (a fraction, say) needs no shaping
    assert build_shaping(None)([0.25, 3.0], Batch()) == [0.25, 3.0]


def test_zscore_standardizes_within_each_group():
    shaping = build_shaping({"type": "zscore"})
    # two groups of two: each is centred on its own mean, so the groups cannot be compared across
    assert shaping([1.0, 3.0, 10.0, 30.0], Batch(group_size=2)) == [-1.0, 1.0, -1.0, 1.0]


def test_zscore_of_a_flat_group_is_zero_not_a_division_by_zero():
    assert build_shaping({"type": "zscore"})([5.0, 5.0], Batch(group_size=2)) == [0.0, 0.0]


def test_build_shaping_rejects_a_bad_spec():
    with pytest.raises(ValueError, match="unknown shaping type"):
        build_shaping({"type": "quadratik", "target": 1})
    with pytest.raises(ValueError, match="needs a type"):
        build_shaping({"target": 1})
    with pytest.raises(ValueError, match="bad parameters"):
        build_shaping({"type": "quadratic"})               # quadratic without a target
    with pytest.raises(ValueError, match="bad parameters"):
        build_shaping({"type": "quadratic", "target": 1, "min": 2})
    with pytest.raises(ValueError, match="unknown shaping type"):
        build_shaping({"type": "hinge"})
