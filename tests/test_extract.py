"""Extract tests: FASTA -> embeddings (mean + per-residue), files written."""

import numpy as np
import pytest
import torch

from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.extract import extract_embeddings, write_embeddings

TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
FASTA = ">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n>B_IDR_2-7\nACDEFGHIKLMN\n"


def _fasta(tmp_path):
    """Write the extraction FASTA fixture and return its path."""
    p = tmp_path / "r.fasta"
    p.write_text(FASTA)
    return p


def test_pool_mean_one_vector_per_sequence(tmp_path):
    """Verify one mean-pooled embedding and metadata row per sequence."""
    model = IDiomTransformer(TINY).eval()
    emb = extract_embeddings(model, _fasta(tmp_path), layers=[1], pool="mean")
    values, index = emb[1]
    assert values.shape == (2, TINY.d_model)
    assert [r["accession"] for r in index] == ["A", "B"]


def test_pool_none_per_residue_with_alignment(tmp_path):
    """Verify per-residue embedding shapes and alignment metadata."""
    model = IDiomTransformer(TINY).eval()
    emb = extract_embeddings(model, _fasta(tmp_path), layers=[0, 1], pool="none", region="all")
    values, index = emb[1]
    assert values.shape[0] == len(index) and values.shape[1] == TINY.d_model
    assert set(index[0]) == {"record_idx", "accession", "source_pos", "residue", "is_idr"}
    assert any(r["is_idr"] for r in index) and set(emb) == {0, 1}


def test_write_embeddings(tmp_path):
    """Verify saved embedding arrays and metadata CSV files."""
    model = IDiomTransformer(TINY).eval()
    emb = extract_embeddings(model, _fasta(tmp_path), layers=[1], pool="mean")
    write_embeddings(emb, tmp_path / "out")
    arr = np.load(tmp_path / "out" / "layer_1.npy")
    assert arr.shape == (2, TINY.d_model)
    assert (tmp_path / "out" / "layer_1_index.csv").exists()


@pytest.mark.parametrize(
    "artifact,flag",
    [
        ("release", "--model"),
        ("hub", "--model"),
        ("checkpoint", "--model"),
        ("checkpoint", "--ckpt"),
    ],
)
def test_extract_cli_loads_supported_artifacts(tmp_path, monkeypatch, artifact, flag):
    """Verify CLI extraction from checkpoints, local releases, and Hub artifacts."""
    from dataclasses import asdict

    from idiom import IDiom
    from idiom.model.extract import main

    model = IDiomTransformer(TINY).eval()
    release = IDiom(model).save_pretrained(tmp_path / "release")
    downloads = []

    def download(repo):
        """Record the requested repository and return the local release fixture."""
        downloads.append(repo)
        return str(release)

    monkeypatch.setattr("idiom.model.io.snapshot_download", download)
    if artifact == "checkpoint":
        path = tmp_path / "model.ckpt"
        torch.save(
            {
                "hyper_parameters": {"model_cfg": asdict(TINY)},
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            },
            path,
        )
    else:
        path = "test/model" if artifact == "hub" else release
    fasta = _fasta(tmp_path)
    out = tmp_path / "out"
    main([flag, str(path), "--fasta", str(fasta), "--layers", "1", "--out", str(out)])
    expected, _ = extract_embeddings(model, fasta, [1])[1]
    np.testing.assert_allclose(np.load(out / "layer_1.npy"), expected, atol=1e-6)
    assert (out / "layer_1_index.csv").is_file()
    assert downloads == (["test/model"] if artifact == "hub" else [])
