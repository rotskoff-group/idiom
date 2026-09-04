"""Tests for the composite (weighted-sum) reward.

The reward is the whole GRPO objective, so the arithmetic is checked against an explicit
from-scratch expectation rather than against itself: every term contributes weight * shaping(reward)
and nothing else does. The cases below cover each way a term can enter or leave the total --
weighted, shaped or raw, zero-weighted, in-process or named by a user module -- because a term that
silently drops out (or is double-counted) changes what the policy optimizes without failing
anywhere. The validation cases matter for the same reason: a term that never runs is a silent
change to the objective, so a malformed one has to fail loudly at build time.
"""

import math

import pytest
from omegaconf import OmegaConf

from idiom.train.grpo.reward import build_reward, quadratic_penalty, register_reward
from idiom.train.grpo.reward import Batch, get_reward, parse_terms


def entropy(idr: str) -> float:
    """The shipped entropy reward, as a scalar, for the explicit-arithmetic expectation below."""
    return get_reward("entropy")([idr], Batch())[0]

GUARDRAILS = [
    {"reward": "entropy", "weight": 0.1, "shaping": {"type": "quadratic", "target": 3.68, "width": 0.2}},
    {"reward": "length", "weight": 0.1, "shaping": {"type": "quadratic", "target": 100, "width": 1.0}},
]


def _cfg(terms, module=None):
    """The configs/grpo.yaml reward structure."""
    return OmegaConf.create({"module": module, "terms": terms})


def test_weighted_sum_matches_explicit_arithmetic():
    register_reward("_r_prol")(lambda idr: idr.count("P") / len(idr) if idr else 0.0)
    register_reward("_r_half")(lambda idr: 0.5)
    cfg = _cfg(GUARDRAILS + [{"reward": "_r_prol", "weight": 2.0},
                             {"reward": "_r_half", "weight": 3.0}])
    idr = "P" * 100  # _r_prol = 1.0, and length sits exactly on the target
    totals, breakdown = build_reward(cfg)([idr], 1)
    expect = (0.1 * quadratic_penalty(entropy(idr), 3.68, 0.2)
              + 0.1 * quadratic_penalty(100.0, 100, 1.0)
              + 2.0 * 1.0
              + 3.0 * 0.5)
    assert math.isclose(totals[0], expect, abs_tol=1e-12)
    # the breakdown names every term, which is what the per-term W&B logging keys off
    assert set(breakdown[0]) == {"entropy", "entropy_raw", "length", "length_raw",
                                 "_r_prol", "_r_prol_raw", "_r_half", "_r_half_raw", "total"}
    assert math.isclose(breakdown[0]["total"], totals[0], abs_tol=1e-12)


def test_breakdown_separates_the_raw_reward_from_the_contribution():
    register_reward("_r_one")(lambda idr: 1.0)
    cfg = _cfg([{"reward": "_r_one", "weight": 2.5}])
    _, breakdown = build_reward(cfg)(["ACDE"], 1)
    assert breakdown[0]["_r_one_raw"] == 1.0   # the raw reward, in its own units
    assert breakdown[0]["_r_one"] == 2.5       # what it contributed to the objective


def test_raw_value_is_the_unshaped_reward():
    # the point of logging both: a length term reads 98 residues while contributing a small penalty
    cfg = _cfg([{"reward": "length", "weight": 1.0,
                 "shaping": {"type": "quadratic", "target": 100, "width": 1.0}}])
    _, breakdown = build_reward(cfg)(["A" * 98], 1)
    assert breakdown[0]["length_raw"] == 98.0
    assert breakdown[0]["length"] == pytest.approx(-0.0004)


def test_shaping_applies_to_an_in_process_reward():
    # shaping applies to an in-process reward exactly like an external one
    register_reward("_r_frac")(lambda idr: 0.30)
    cfg = _cfg([{"reward": "_r_frac", "weight": 1.0,
                 "shaping": {"type": "quadratic", "target": 0.15, "width": 1.0}}])
    totals, _ = build_reward(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], quadratic_penalty(0.30, 0.15, 1.0), abs_tol=1e-12)


def test_a_reward_without_shaping_is_used_raw():
    register_reward("_r_frac2")(lambda idr: 0.7)
    totals, _ = build_reward(_cfg([{"reward": "_r_frac2", "weight": 2.0}]))(["ACDE"], 1)
    assert math.isclose(totals[0], 1.4, abs_tol=1e-12)


def test_zero_weight_term_is_logged_but_not_optimized():
    register_reward("_watch")(lambda idr: 7.0)
    cfg = _cfg([{"reward": "_watch", "weight": 0.0}])
    totals, breakdown = build_reward(cfg)(["AA"], 1)
    assert totals == [0.0]                       # contributes nothing to the objective
    assert breakdown[0]["_watch_raw"] == 7.0     # and is still scored and logged


def test_disabled_term_is_skipped_whole():
    register_reward("_r_off")(lambda idr: 99.0)
    cfg = _cfg([{"reward": "_r_off", "weight": 1.0, "enabled": False}])
    totals, breakdown = build_reward(cfg)(["ACDE"], 1)
    assert totals == [0.0] and breakdown[0] == {"total": 0.0}   # not run, not logged


def test_a_disabled_term_is_not_validated():
    # the shipped config carries a menu of switched-off terms, so one may name a reward this
    # environment cannot import, or a module that is not installed; neither may break the run
    cfg = _cfg([{"reward": "length", "weight": 1.0},
                {"enabled": False, "reward": "nothing_registers_this", "weight": 1.0},
                {"enabled": False, "module": "no.such.module", "cmd": "false", "label": "x"}])
    totals, _ = build_reward(cfg)(["AAA"], 1)
    assert totals == [3.0]


def test_enabled_and_zero_weight_differ():
    # weight 0 runs the term and logs it; enabled false does neither
    register_reward("_r_seven")(lambda idr: 7.0)
    _, watched = build_reward(_cfg([{"reward": "_r_seven", "weight": 0.0}]))(["AA"], 1)
    _, skipped = build_reward(_cfg([{"reward": "_r_seven", "enabled": False}]))(["AA"], 1)
    assert watched[0]["_r_seven_raw"] == 7.0
    assert "_r_seven_raw" not in skipped[0]


def test_no_terms_scores_zero():
    totals, breakdown = build_reward(_cfg([]))(["ACDE"], 1)
    assert totals == [0.0] and breakdown[0] == {"total": 0.0}


def test_the_same_reward_can_appear_twice_under_distinct_labels():
    # e.g. one term uses the raw value while another shapes or weights it differently
    register_reward("_r_dup")(lambda idr: 1.0)
    cfg = _cfg([{"reward": "_r_dup", "weight": 1.0},
                {"reward": "_r_dup", "label": "_r_dup_scaled", "weight": 2.0}])
    totals, breakdown = build_reward(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 3.0, abs_tol=1e-12)
    assert breakdown[0]["_r_dup"] == 1.0 and breakdown[0]["_r_dup_scaled"] == 2.0


def test_batched_reward_sees_the_whole_batch():
    register_reward("_r_batched", batched=True)(
        lambda idrs, batch: [float(len(idrs) * batch.group_size)] * len(idrs))
    totals, _ = build_reward(_cfg([{"reward": "_r_batched", "weight": 1.0}]))(["A", "B"], 2)
    assert totals == [4.0, 4.0]


def test_module_is_imported_before_lookup(tmp_path):
    # a user drops a *.py with @register_reward; module imports it so the name resolves
    mod = tmp_path / "myreward.py"
    mod.write_text(
        "from idiom.train.grpo.reward import register_reward\n"
        "register_reward('_from_module')(lambda idr: 4.0)\n"
    )
    cfg = _cfg([{"reward": "_from_module", "weight": 0.5, "module": str(mod)}])
    totals, _ = build_reward(cfg)(["ACDE"], 1)
    assert math.isclose(totals[0], 2.0, abs_tol=1e-12)


def test_scores_stay_aligned_across_a_batch():
    # totals[i] must correspond to idrs[i]; a term that reorders would corrupt the advantages
    cfg = _cfg([{"reward": "length", "weight": 1.0}])
    idrs = ["A", "AA", "AAA", ""]
    totals, breakdown = build_reward(cfg)(idrs, 2)
    assert totals == [1.0, 2.0, 3.0, 0.0]
    assert [b["total"] for b in breakdown] == totals


def test_config_errors_are_raised_at_build_time():
    with pytest.raises(KeyError, match="unknown reward"):
        build_reward(_cfg([{"reward": "not_a_reward", "weight": 1.0}]))
    with pytest.raises(ValueError, match="exactly one of reward"):
        build_reward(_cfg([{"weight": 1.0}]))
    with pytest.raises(ValueError, match="exactly one of reward"):
        build_reward(_cfg([{"reward": "length", "cmd": "true", "weight": 1.0}]))
    with pytest.raises(ValueError, match="needs a label"):
        build_reward(_cfg([{"cmd": "true", "weight": 1.0}]))
    with pytest.raises(ValueError, match="duplicate label"):
        build_reward(_cfg([{"reward": "length"}, {"reward": "length"}]))
    with pytest.raises(ValueError, match="unknown key"):
        # the old config spelled a target on the term itself; it must not be silently ignored
        build_reward(_cfg([{"reward": "length", "weight": 1.0, "target": 100}]))
    with pytest.raises(ValueError, match="unknown shaping type"):
        build_reward(_cfg([{"reward": "length", "shaping": {"type": "quadratik", "target": 1}}]))


# --- reward: "module:function" -------------------------------------------------------------
# The path for code that already exists in the environment IDiom was installed into: a term names
# an importable callable, so nothing has to be decorated, registered, or copied into this repo.


@pytest.fixture
def lab_package(tmp_path, monkeypatch):
    """A stand-in for a package installed beside IDiom, importing nothing from it."""
    pkg = tmp_path / "labpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "scoring.py").write_text(
        "def score_idr(seq):\n"
        "    return seq.count('W') / len(seq) if seq else 0.0\n"
        "\n"
        "def score_batch(seqs, batch):\n"
        "    return [float(len(s)) for s in seqs]\n"
        "\n"
        "NOT_CALLABLE = 3\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return pkg


def test_callable_reward_needs_no_decorator(lab_package):
    totals, breakdown = build_reward(_cfg([{"reward": "labpkg.scoring:score_idr", "weight": 2.0}]))(
        ["WWAA", "AAAA"], 1)
    assert totals == [1.0, 0.0]  # 2/4 aromatic * weight 2, and 0
    # logged under the function's own name, not the whole dotted spec
    assert set(breakdown[0]) == {"score_idr", "score_idr_raw", "total"}


def test_callable_reward_accepts_a_file_path(lab_package):
    spec = f"{lab_package / 'scoring.py'}:score_idr"
    assert build_reward(_cfg([{"reward": spec, "weight": 1.0}]))(["WWAA"], 1)[0] == [0.5]


def test_callable_reward_batched_form(lab_package):
    cfg = _cfg([{"reward": "labpkg.scoring:score_batch", "batched": True, "weight": 1.0}])
    assert build_reward(cfg)(["WWAA", "AAA"], 1)[0] == [4.0, 3.0]


def test_callable_reward_label_can_be_overridden(lab_package):
    cfg = _cfg([{"reward": "labpkg.scoring:score_idr", "label": "aromatic", "weight": 1.0}])
    _, breakdown = build_reward(cfg)(["WWAA"], 1)
    assert set(breakdown[0]) == {"aromatic", "aromatic_raw", "total"}


@pytest.mark.parametrize("spec, match", [
    ("labpkg.nope:score_idr", "cannot import"),
    ("labpkg.scoring:nope", "has no attribute"),
    ("labpkg.scoring:NOT_CALLABLE", "not callable"),
])
def test_bad_callable_reward_fails_at_build_time(lab_package, spec, match):
    # a term that cannot run must fail while the config is parsed, not on the first training step
    with pytest.raises(ValueError, match=match):
        build_reward(_cfg([{"reward": spec, "weight": 1.0}]))


def test_batched_is_rejected_on_a_registered_reward():
    with pytest.raises(ValueError, match="batched applies only"):
        build_reward(_cfg([{"reward": "entropy", "batched": True, "weight": 1.0}]))


# --- reward.add: choosing the objective at launch --------------------------------------------
# The config carries the guardrails and a named menu; a run says what it optimizes on the command
# line. Names, not list positions, because positions shift whenever the menu grows.

MENU = {"rg": {"cmd": "true", "label": "rg", "weight": 0.5,
               "shaping": {"type": "quadratic", "target": 25, "width": 0.2}},
        "prol": {"reward": "fraction_proline", "weight": 1.0}}


def _addcfg(add):
    return OmegaConf.create({"module": None, "terms": GUARDRAILS, "presets": MENU, "add": add})


def test_add_is_empty_by_default():
    assert [s.label for s in parse_terms(_addcfg([]))] == ["entropy", "length"]


def test_add_appends_a_named_preset_after_the_guardrails():
    assert [s.label for s in parse_terms(_addcfg(["prol"]))] == ["entropy", "length",
                                                                 "fraction_proline"]


def test_add_preserves_the_order_it_was_named_in():
    assert [s.label for s in parse_terms(_addcfg(["rg", "prol"]))][2:] == ["rg", "fraction_proline"]


def test_add_accepts_a_whole_term_inline():
    term = {"reward": "fraction_proline", "label": "mine", "weight": 3.0}
    specs = parse_terms(_addcfg([term]))
    assert specs[-1].label == "mine" and specs[-1].weight == 3.0


def test_add_carries_the_presets_shaping_and_weight():
    spec = parse_terms(_addcfg(["rg"]))[-1]
    assert spec.weight == 0.5 and spec.shaping["target"] == 25


def test_unknown_preset_names_the_menu():
    with pytest.raises(ValueError, match=r"unknown preset 'nope'.*\['prol', 'rg'\]"):
        parse_terms(_addcfg(["nope"]))


def test_add_rejects_a_non_name_non_mapping():
    with pytest.raises(ValueError, match="expected a preset name or a term mapping"):
        parse_terms(_addcfg([3]))


def test_an_unused_preset_is_never_validated():
    # the menu may name a reward this environment cannot import; that is what makes it free
    cfg = OmegaConf.create({"module": None, "terms": GUARDRAILS,
                            "presets": {"bad": {"reward": "no.such.module:f", "weight": 1.0}},
                            "add": []})
    assert [s.label for s in parse_terms(cfg)] == ["entropy", "length"]
