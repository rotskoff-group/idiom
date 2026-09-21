"""Tests for SAE feature enrichment."""

import json
import math

import numpy as np

from idiom.sae.features.enrichment import (
    bh_fdr,
    boundary_features,
    enrich,
    enriched_mask,
    feature_counts,
    length_match,
    load_sequences,
    select_features,
)
from idiom.sae.features.signatures import write_signature


def _make_dataset(tmp_path, per_seq_features, num_latents=10, n_res=5, values=None):
    """Write a synthetic feature dataset: per_seq_features[s] = [k features] for every residue."""
    d = tmp_path / "fd"
    d.mkdir(exist_ok=True)
    k = len(per_seq_features[0])
    ti, si, pi, tv, strings = [], [], [], [], []
    for s, feats in enumerate(per_seq_features):
        strings.append("132" + "A" * n_res)
        for r in range(n_res):
            ti.append(list(feats))
            tv.append(values[s][r] if values else [1.0] * k)
            si.append(s)
            pi.append(3 + r)
    np.save(d / "top_indices.npy", np.array(ti, dtype=np.int32))
    np.save(d / "top_values.npy", np.array(tv, dtype=np.float32))
    np.save(d / "seq_idx.npy", np.array(si, dtype=np.int32))
    np.save(d / "pos_idx.npy", np.array(pi, dtype=np.int32))
    (d / "strings.json").write_text(json.dumps(strings))
    (d / "meta.json").write_text(json.dumps({"num_latents": num_latents, "k": k}))
    return d


def test_feature_counts(tmp_path):
    """Verify per-feature sequence counts from the feature dataset."""
    d = _make_dataset(tmp_path, [[1, 2], [1, 3]])
    counts, n_seq = feature_counts(d)
    assert n_seq == 2
    assert counts[1] == 2  # fires in both sequences (max-pooled, not counted per residue)
    assert counts[2] == 1
    assert counts[3] == 1
    assert counts[0] == 0
    counts, n_seq = feature_counts(d, keep=[0])
    assert n_seq == 1 and counts[1] == 1 and counts[3] == 0


def test_bh_fdr_monotone_and_bounded():
    """Verify monotonic, bounded FDR values for sorted input p-values."""
    q = bh_fdr(np.array([0.001, 0.01, 0.5, 0.9]))
    assert np.all((q >= 0) & (q <= 1))
    assert np.all(np.diff(q) >= -1e-12)
    assert q[0] < q[-1]


def test_feature_counts_ignore_zero_selections_and_keep_silent_sequences(tmp_path):
    """Verify that zero activations are excluded while silent sequences remain counted."""
    d = _make_dataset(
        tmp_path, [[1, 2], [1, 3]], n_res=2, values=[[[1.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    strings = json.loads((d / "strings.json").read_text())
    (d / "strings.json").write_text(json.dumps([*strings, "132"]))
    counts, n = feature_counts(d)
    assert n == 3 and counts[1] == 1 and counts.sum() == 1
    counts, n = feature_counts(d, keep=[1, 2])
    assert n == 2 and counts.sum() == 0
    counts, n = feature_counts(d, keep=[2])
    assert n == 1 and counts.sum() == 0


def test_two_sided_p_matches_normal():
    """Verify two-sided normal p-values at known reference points."""
    from idiom.sae.features.enrichment import _two_sided_p

    # erfc(|z|/sqrt2) == 2 * normal survival function
    assert math.isclose(float(_two_sided_p(np.array([0.0]))[0]), 1.0, abs_tol=1e-12)
    assert math.isclose(float(_two_sided_p(np.array([1.959964]))[0]), 0.05, abs_tol=1e-5)


def test_enrich_separates_signal_from_noise():
    """Verify enrichment of a strong feature and rejection of background features."""
    n_latents = 4
    n_pos, n_neg = 100, 1000
    # Feature 0 is enriched, feature 1 is neutral, and feature 2 has too few firings to test
    a = np.array([100.0, 50.0, 1.0, 0.0])
    b = np.array([0.0, 500.0, 0.0, 0.0])
    r = enrich(a, n_pos, b, n_neg, n_latents)

    assert r["log2or"][0] > 5 and r["z"][0] > 5 and r["fdr"][0] < 1e-3
    assert abs(r["log2or"][1]) < 0.5 and r["fdr"][1] > 1e-3
    assert not r["active"][2]  # pooled firing count below MIN_TOTAL_FIRE
    m = enriched_mask(r)
    assert m[0] and not m[1] and not m[2]
    assert r["prev_pos"][0] == 1.0 and r["prev_neg"][0] == 0.0


def test_select_features_ranks_by_log2or(tmp_path):
    """Verify that selected features are ordered by descending log odds ratio."""
    n_latents = 5
    a = np.array([90.0, 100.0, 60.0, 0.0, 0.0])
    b = np.array([10.0, 300.0, 5.0, 0.0, 0.0])
    r = enrich(a, 100, b, 1000, n_latents)
    ids = select_features(r, n=2, drop_boundary=False)["ids"]
    assert len(ids) == 2
    assert r["log2or"][ids[0]] >= r["log2or"][ids[1]]


def test_boundary_features_flags_terminal_firing(tmp_path):
    # Residues occupy FIM positions 3..17; edge=2 selects positions <=5 or >=15
    """Verify that boundary filtering flags terminal peaks but retains interior peaks."""
    n_res = 15
    per_seq = [[7, 5]] * 4
    vals = []
    for _ in range(4):
        # feature 7 strongest at r=0 (the excision boundary), feature 5 strongest at r=7 (middle)
        vals.append([[10.0 if r == 0 else 0.1, 10.0 if r == 7 else 0.1] for r in range(n_res)])
    d = _make_dataset(tmp_path, per_seq, num_latents=10, n_res=n_res, values=vals)

    flagged = boundary_features(d, [7, 5], top_windows=4)
    assert 7 in flagged, "a feature peaking at the IDR edge must be flagged"
    assert 5 not in flagged, "a feature peaking mid-sequence must not be flagged"


def test_write_signature_roundtrip(tmp_path):
    """Verify that signature updates preserve other cases and provenance."""
    p = tmp_path / "sig.json"
    write_signature(p, {"my_set": [3, 1, 2]}, case="top30", provenance={"sae": "test"})
    write_signature(p, {"my_set": [3]}, case="private30")
    blob = json.loads(p.read_text())
    assert blob["top30"]["my_set"] == [3, 1, 2]  # order preserved (rank order matters)
    assert blob["private30"]["my_set"] == [3]
    assert blob["_provenance"]["sae"] == "test"


def test_load_sequences_reads_spans_and_falls_back_to_whole_sequence(tmp_path):
    """Verify parsed IDR spans and whole-sequence fallback for unannotated inputs."""
    fa = tmp_path / "in.fasta"
    fa.write_text(">P1_IDR_2-5\nACDEFGHI\n>plain some description here\nMKVGSDEQ\n")
    recs = load_sequences(fa)
    # the header's 1-based inclusive span becomes a 0-based half-open one
    assert [(r.accession, r.idr_start, r.idr_end) for r in recs] == [("P1", 1, 5), ("plain", 0, 8)]


def test_load_sequences_ignores_an_out_of_range_span(tmp_path):
    """Reject out-of-range spans rather than interpreting them as whole IDRs."""
    fa = tmp_path / "bad.fasta"
    fa.write_text(">P2_IDR_0-999\nACDE\n")
    assert load_sequences(fa) == []


def _rec(acc, length, start=0):
    """Create an alanine record with the requested IDR length and starting offset."""
    from idiom.data.records import Record

    return Record(acc, "A" * (start + length), start, start + length)


def test_length_match_follows_the_positive_length_distribution():
    # Length matching should prevent length-sensitive features from appearing enriched
    """Verify that background sampling follows the positive-set length distribution."""
    rng = np.random.default_rng(0)
    positives = [_rec(f"p{i}", 10) for i in range(20)]
    background = [_rec(f"s{i}", 10) for i in range(100)] + [_rec(f"l{i}", 300) for i in range(100)]
    picked = length_match(positives, background, n=40, rng=rng)
    assert len(picked) == 40
    assert all(r.accession.startswith("s") for r in picked)


def test_length_match_tops_up_when_a_bin_cannot_be_filled():
    # Underfilled length bins must draw from the remaining pool
    """Verify that other length bins fill a sampling shortfall."""
    rng = np.random.default_rng(0)
    positives = [_rec(f"p{i}", 10) for i in range(20)]
    background = [_rec(f"s{i}", 10) for i in range(5)] + [_rec(f"l{i}", 300) for i in range(100)]
    picked = length_match(positives, background, n=40, rng=rng)
    assert len(picked) == 40
    assert sum(r.accession.startswith("s") for r in picked) == 5
