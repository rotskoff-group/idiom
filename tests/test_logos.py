"""Tests for the residue windows a feature logo is built from."""

import numpy as np

from idiom.sae.features import per_sequence_activations, top_windows


def _index(seqs):
    """Per-row metadata for a set of sequences, in the order encode(pool='none') returns rows."""
    return [{"accession": acc, "source_pos": i, "residue": r} for acc, seq in seqs for i, r in enumerate(seq)]


def test_per_sequence_activations_regroups_rows_by_accession():
    """Verify legacy grouping of feature rows by accession."""
    seqs = [("A", "MKV"), ("B", "GSGS")]
    per_seq = per_sequence_activations(np.zeros((7, 4)), _index(seqs))
    assert [s for s, _ in per_seq] == ["MKV", "GSGS"]
    assert [idx.tolist() for _, idx in per_seq] == [[0, 1, 2], [3, 4, 5, 6]]


def test_per_sequence_activations_orders_rows_by_source_position():
    # Encoder rows may be shuffled; logo windows must follow source-residue order
    """Verify that feature rows are restored to original protein order."""
    index = [
        {"accession": "A", "source_pos": 2, "residue": "V"},
        {"accession": "A", "source_pos": 0, "residue": "M"},
        {"accession": "A", "source_pos": 1, "residue": "K"},
    ]
    ((residues, idx),) = per_sequence_activations(np.zeros((3, 2)), index)
    assert residues == "MKV" and idx.tolist() == [1, 2, 0]


def _feats(n_res, peaks):
    """Activations for one feature (column 0), with peaks as {row: value}."""
    f = np.zeros((n_res, 1))
    for row, val in peaks.items():
        f[row, 0] = val
    return f


def test_top_windows_centres_on_the_peak_and_ranks_by_activation():
    """Verify peak-centered windows ranked by activation strength."""
    seqs = [("A", "AAAAKWAAAA"), ("B", "CCCCPYCCCC")]  # each peaks on its 6th residue
    feats = _feats(20, {5: 1.0, 15: 9.0})  # B activates harder
    per_seq = per_sequence_activations(feats, _index(seqs))
    # 5-residue windows centred on the peak residue (W, Y), most-active sequence first
    assert top_windows(0, feats, per_seq, half_width=2) == ["CPYCC", "AKWAA"]


def test_top_windows_clamps_a_peak_at_the_edge():
    """Verify that windows around edge peaks remain within sequence bounds."""
    seqs = [("A", "WAAAA")]
    feats = _feats(5, {0: 1.0})
    per_seq = per_sequence_activations(feats, _index(seqs))
    assert top_windows(0, feats, per_seq, half_width=1) == ["WAA"]  # clamped, still full width


def test_top_windows_skips_short_sequences_and_silent_features():
    """Verify exclusion of short sequences and inactive features from logo windows."""
    seqs = [("short", "AA"), ("silent", "CCCCCCC")]
    feats = _feats(9, {})
    per_seq = per_sequence_activations(feats, _index(seqs))
    assert top_windows(0, feats, per_seq, half_width=2) == []
