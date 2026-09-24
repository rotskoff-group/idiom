"""Peak-window alignment and information content from saved sparse activations."""

import json

import numpy as np

from idiom.sae.features import FeatureDataset, feature_windows, logo_data


def test_windows_keep_peaks_aligned_and_missing_positions(tmp_path):
    """Edge peaks and short sequences contribute at their real relative offsets."""
    np.save(tmp_path / "top_indices.npy", np.zeros((5, 1), dtype=np.int32))
    np.save(tmp_path / "top_values.npy", np.array([[3], [1], [0], [1], [2]], dtype=np.float32))
    np.save(tmp_path / "seq_idx.npy", np.array([0, 0, 0, 1, 1]))
    np.save(tmp_path / "pos_idx.npy", np.array([3, 4, 5, 3, 4]))
    (tmp_path / "strings.json").write_text(json.dumps(["132ACD", "132EF"]))
    (tmp_path / "meta.json").write_text(json.dumps(dict(k=1, num_latents=2, layer=0)))
    fd = FeatureDataset(tmp_path)
    windows = feature_windows(fd, 0, half_width=2)
    assert [w.residues for w in windows] == ["--ACD", "-EF--"]
    assert [w.peak_position for w in windows] == [3, 4]
    assert np.isnan(windows[0].activations[:2]).all()
    data = logo_data(fd, 0, half_width=2)
    assert data["counts"].sum(axis=1).tolist() == [0, 1, 2, 1, 1]
    assert data["information"].shape == (5, 20)
    assert not data["information"][0].any()
    np.testing.assert_allclose(data["mean_activation"], [0, 0.5, 2.5, 0.5, 0])
    empty = logo_data(fd, 1, half_width=2)
    assert not empty["windows"] and not empty["counts"].any()
    assert feature_windows(fd, 0, n=0) == []
