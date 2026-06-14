"""P5 feature-dataset tests (CPU-only): build -> read -> reduce, with residue alignment."""

from idiom.analysis.build_feature_dataset import build_feature_dataset
from idiom.analysis.feature_activations import (
    FeatureDataset,
    feature_trace_for_sequence,
    top_n_sequences,
)
from idiom.data.io import Record
from idiom.data.tokenizer import RESIDUES, Tokenizer
from idiom.model import IDiomTransformer, ModelConfig
from idiom.sae import SparseCoder

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
RECS = [Record(f"r{i}", "MEDSKVDNRPQACDEFG", 3, 12) for i in range(5)]


def _build(tmp_path):
    model = IDiomTransformer(TINY)
    sae = SparseCoder(TINY.d_model, num_latents=32, k=4)
    out = build_feature_dataset(model, sae, RECS, layer=1, out_dir=tmp_path / "fd", batch_size=2)
    return out, sae


def test_build_and_read(tmp_path):
    out, sae = _build(tmp_path)
    for f in ("top_indices.npy", "top_values.npy", "seq_idx.npy", "pos_idx.npy", "strings.json", "meta.json"):
        assert (out / f).exists()
    fd = FeatureDataset(out, in_memory=True)
    assert fd.k == int(sae.k) and fd.num_latents == sae.num_latents and fd.layer == 1
    assert fd.top_indices.shape[1] == fd.k
    assert len(fd.strings) == len(RECS)
    assert fd.sequence(0).startswith("1")  # FIM string


def test_pos_idx_aligns_to_residues(tmp_path):
    out, _ = _build(tmp_path)
    fd = FeatureDataset(out, in_memory=True)
    # every (seq_idx, pos_idx) must point at a real residue char in that FIM string.
    for row in range(min(20, len(fd.seq_idx))):
        s = fd.sequence(int(fd.seq_idx[row]))
        assert s[int(fd.pos_idx[row])] in RESIDUES


def test_reductions_run(tmp_path):
    out, _ = _build(tmp_path)
    fd = FeatureDataset(out, in_memory=True)
    feat = int(fd.top_indices[0, 0])  # a feature that fired somewhere
    seqs, scores = top_n_sequences(fd, feat, n=3, sort_by="peak")
    assert len(seqs) >= 1 and (scores > 0).all()
    pos, acts = feature_trace_for_sequence(fd, int(seqs[0]), feat)
    assert len(pos) == len(acts)
