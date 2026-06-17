"""P5 extract tests (CPU-only): FASTA -> embeddings (mean + per-residue), files written."""

import numpy as np

from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.extract import embed_fasta, write_embeddings

TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
FASTA = ">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n>B_IDR_2-7\nACDEFGHIKLMN\n"


def _fasta(tmp_path):
    p = tmp_path / "r.fasta"
    p.write_text(FASTA)
    return p


def test_pool_mean_one_vector_per_sequence(tmp_path):
    model = IDiomTransformer(TINY).eval()
    emb = embed_fasta(model, _fasta(tmp_path), layers=[1], pool="mean")
    values, index = emb[1]
    assert values.shape == (2, TINY.d_model)  # 2 sequences -> 2 vectors
    assert [r["accession"] for r in index] == ["A", "B"]


def test_pool_none_per_residue_with_alignment(tmp_path):
    model = IDiomTransformer(TINY).eval()
    emb = embed_fasta(model, _fasta(tmp_path), layers=[0, 1], pool="none")
    values, index = emb[1]
    # one row per residue across both sequences; alignment metadata present.
    assert values.shape[0] == len(index) and values.shape[1] == TINY.d_model
    assert set(index[0]) == {"accession", "source_pos", "residue", "is_idr"}
    assert any(r["is_idr"] for r in index) and set(emb) == {0, 1}


def test_write_embeddings(tmp_path):
    model = IDiomTransformer(TINY).eval()
    emb = embed_fasta(model, _fasta(tmp_path), layers=[1], pool="mean")
    write_embeddings(emb, tmp_path / "out")
    arr = np.load(tmp_path / "out" / "layer_1.npy")
    assert arr.shape == (2, TINY.d_model)
    assert (tmp_path / "out" / "layer_1_index.csv").exists()
