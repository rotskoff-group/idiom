"""Regression tests for release-facing feature analysis and signature interchange."""

import json

import numpy as np
import pytest

from idiom.sae.features import (
    FeatureDataset,
    combine_signatures,
    load_enrichment,
    load_signatures,
    save_enrichment,
    select_features,
    write_signature,
)
from idiom.sae.features.enrichment import boundary_features, feature_counts, length_match


def dataset(path, values=None):
    """A short IDR with an interior firing and silent edge selections."""
    path.mkdir()
    values = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32) if values is None else values
    n = len(values)
    np.save(path / "top_indices.npy", np.zeros((n, 1), dtype=np.int32))
    np.save(path / "top_values.npy", values[:, None])
    np.save(path / "seq_idx.npy", np.zeros(n, dtype=np.int32))
    np.save(path / "pos_idx.npy", np.arange(3, n + 3))
    (path / "strings.json").write_text(json.dumps(["132" + "A" * 9]))
    (path / "meta.json").write_text(json.dumps(dict(k=1, num_latents=3, layer=0, fim_mode="unprompted")))
    return path


def test_boundary_ignores_silent_edges_and_empty_rows(tmp_path):
    path = dataset(tmp_path / "fd")
    assert boundary_features(path, [0]) == set()
    empty = dataset(tmp_path / "empty", np.zeros(0, dtype=np.float32))
    assert boundary_features(empty, [0]) == set()
    fd = FeatureDataset(empty)
    assert fd.top_sequences(0)[0].size == 0
    assert fd.trace(0, 0)[0].size == 0
    assert feature_counts(empty)[1] == 1


def test_dataset_rejects_bad_indices_and_metadata(tmp_path):
    path = dataset(tmp_path / "fd")
    fd = FeatureDataset(path)
    assert fd.fim_mode == "unprompted" and fd._order is None
    for fn in (fd.row_activations, fd.feature_stats):
        with pytest.raises(ValueError):
            fn(-1)
    with pytest.raises(ValueError):
        fd.trace(1, 0)
    with pytest.raises(ValueError):
        feature_counts(fd, keep=[3])
    np.save(path / "pos_idx.npy", np.full(9, 100))
    with pytest.raises(ValueError, match="Position"):
        FeatureDataset(path)


def test_selection_ties_prevalence_and_result_roundtrip(tmp_path):
    result = dict(
        log2or=np.array([3.0, 3.0, 2.0]),
        prev_pos=np.array([0.02, 0.03, 0.5]),
        fdr=np.array([0.0001] * 3),
        n_pos=100,
        n_neg=200,
    )
    selection = select_features(result, n=2, drop_boundary=False, prev_pos_floor=0.01)
    assert selection["ids"] == [0, 1]
    save_enrichment(tmp_path / "result.npz", result, selection)
    restored, selected = load_enrichment(tmp_path / "result.npz")
    assert selected["ids"] == [0, 1] and restored["n_pos"] == 100
    np.testing.assert_array_equal(restored["log2or"], result["log2or"])
    assert select_features(result, n=0, drop_boundary=False)["ids"] == []


def test_length_matching_exact_cap_and_no_replacement():
    from idiom.data.records import Record

    positives = [Record(str(n), "A" * n, 0, n) for n in [10, 30, 50]]
    background = [Record(str(i), "A" * 10, 0, 10) for i in range(20)]
    for n in (0, 1, 2, 5, 40):
        picked = length_match(positives, background, n=n, rng=np.random.default_rng(0))
        assert len(picked) == min(n, len(background))
        assert len({r.accession for r in picked}) == len(picked)


def test_signatures_identity_union_and_legacy(tmp_path):
    path = write_signature(
        tmp_path / "signature.json", {"a": [2, 1], "b": [1, 0]}, provenance={"sae": "lens"}
    )
    sets = load_signatures(path, sae="lens", num_latents=3)
    assert combine_signatures(sets) == [2, 1, 0]
    assert sets == {"a": [2, 1], "b": [1, 0]}
    with pytest.raises(ValueError, match="does not match"):
        load_signatures(path, sae="other")
    with pytest.raises(ValueError, match="different SAEs"):
        write_signature(path, {"c": [1]}, provenance={"sae": "other"})
    for ids in ([], [-1], [3], [1, 1], [0.5]):
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({"top30": {"a": ids}}))
        with pytest.raises(ValueError):
            load_signatures(legacy, num_latents=3)
    legacy.write_text(json.dumps({"top30": {"a": [1]}}))
    assert load_signatures(legacy, sae="any", num_latents=3) == {"a": [1]}


def test_builder_validates_before_inference(tmp_path, monkeypatch):
    from idiom.data.records import Record
    from idiom.model import IDiomTransformer, ModelConfig
    from idiom.sae import SparseCoder
    from idiom.sae.features import build_feature_dataset

    model = IDiomTransformer(ModelConfig(n_layers=1, d_model=8, n_heads=2, max_seq_len=16))
    sae = SparseCoder(8, num_latents=4, k=1)
    monkeypatch.setattr(model, "forward", lambda *a, **k: pytest.fail("invalid input reached inference"))
    record = Record("valid", "AAA", 0, 3)
    for records, batch_size in (([], 1), ([record], 0), ([Record("long", "A" * 20, 0, 20)], 1)):
        with pytest.raises(ValueError):
            build_feature_dataset(model, sae, records, 0, tmp_path / "fd", batch_size=batch_size)
        assert not (tmp_path / "fd").exists()


def test_signature_null_provenance_preserves_identity(tmp_path):
    """A null update must not allow old signatures to be relabeled with another SAE."""
    path = write_signature(tmp_path / "sig.json", {"first": [1]}, provenance={"sae": "A"})
    write_signature(path, {"second": [2]}, case="second", provenance={"sae": None})
    assert load_signatures(path, sae="A") == {"first": [1]}
    with pytest.raises(ValueError, match="different SAEs"):
        write_signature(path, {"third": [3]}, case="third", provenance={"sae": "B"})
