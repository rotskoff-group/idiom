"""Pure-function tests for the eval suite (no model/GPU)."""

import numpy as np

from eval.distances import w1, w1_normalized, w1_table
from eval.metrics import FEATURES, features_table, sequence_features, summarize
from eval.validity import validity_stats


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
