"""Pure-function tests for the eval suite (no model/GPU)."""

import numpy as np

from extras.eval.distances import w1, w1_normalized, w1_table
from extras.eval.localization import COMPARTMENTS, localization_stats
from extras.eval.metrics import (
    FEATURES,
    aa_entropy,
    composition_extras_table,
    features_table,
    lcd_fraction,
    sequence_features,
    summarize,
)
from extras.eval.validity import validity_stats


def test_sequence_features():
    f = sequence_features("MEEDKKRRSSAAPPPGGYYFF")
    assert set(f) == set(FEATURES)
    assert not np.isnan(f["FCR"]) and 0 <= f["FCR"] <= 1


def test_features_table_and_summary():
    tab = features_table(["MEEDKKRR", "AAAAGGGG", "SSSSPPPP"])
    assert tab["FCR"].shape == (3,)
    s = summarize(tab)
    assert s["FCR"]["n"] == 3 and "mean" in s["FCR"]


def test_w1():
    a, b = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.2, 0.3, 0.4, 0.5])
    assert abs(w1(a, b) - 0.1) < 1e-9
    assert w1_normalized(a, b) > 0
    assert np.isnan(w1(np.array([]), np.array([1.0])))  # empty -> nan


def test_w1_table():
    wt = w1_table(features_table(["MEEDKK", "RRSSAA"]), features_table(["MKKEED", "SSAARR"]))
    assert "FCR" in wt and "w1_norm" in wt["FCR"]


def test_validity():
    v = validity_stats(["AAAA", "BBBBBBBB", "CCCC"], max_new_tokens=8)
    assert v["frac_terminated"] == 2 / 3  # the length-8 seq hit the cap
    assert v["n"] == 3 and v["frac_unique"] == 1.0


def test_aa_entropy_and_lcd():
    assert aa_entropy("AAAA") == 0.0  # single residue -> zero entropy
    assert abs(aa_entropy("ACDEFGHIKLMNPQRSTVWY") - np.log2(20)) < 1e-9  # uniform 20 -> max
    assert np.isnan(aa_entropy(""))
    assert lcd_fraction("RGRG") == 1.0 and lcd_fraction("AAAA") == 0.0
    assert lcd_fraction("RGAA") == 0.5
    tab = composition_extras_table(["AAAA", "RGRG"])
    assert tab["aa_entropy"].shape == (2,) and tab["lcd_fraction"][1] == 1.0


def test_localization_stats():
    # two seqs: #0 argmax Nucleus, #1 argmax Cytoplasm
    scores = {c: np.zeros(2) for c in COMPARTMENTS}
    scores["Nucleus"] = np.array([0.9, 0.1])
    scores["Cytoplasm"] = np.array([0.2, 0.8])
    st = localization_stats(scores, expected="Nucleus")
    assert st["n"] == 2
    assert abs(st["mean_expected"] - 0.5) < 1e-9
    assert st["frac_argmax_expected"] == 0.5
