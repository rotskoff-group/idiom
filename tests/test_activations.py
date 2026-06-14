"""P2 extractor tests (CPU-only): residue-only selection, marker drop, alignment, multi-layer."""

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.activations import extract_activations

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=3, d_model=32, n_heads=4, max_seq_len=32)


def _tokens():
    # [START, '1', A, C, '3', D, '2', E, PAD] — markers/controls interleaved with 4 residues.
    ids = [TOK.start_id, *TOK.encode("1AC3D2E"), TOK.pad_id]
    return torch.tensor(ids).unsqueeze(0)  # [1, 9]


def test_residue_only_selection_and_alignment():
    model = IDiomTransformer(TINY).eval()
    tokens = _tokens()
    out = extract_activations(model, tokens, layers=[1])
    act = out[1]
    # 4 residues (A, C, D, E); markers '1','3','2', START, PAD all dropped.
    assert act.values.shape == (4, TINY.d_model)
    assert act.pos_idx.tolist() == [2, 3, 5, 7]  # positions of A, C, D, E in the sequence
    assert act.token_id.tolist() == TOK.encode("ACDE")
    assert "".join(TOK.decode([i]) for i in act.token_id.tolist()) == "ACDE"
    assert (act.token_id < TOK.n_residues).all()  # all are residues


def test_keep_markers_option():
    model = IDiomTransformer(TINY).eval()
    out = extract_activations(model, _tokens(), layers=[0], drop_markers=False)
    # residues (4) + FIM markers (3) = 7; START + PAD still dropped.
    assert out[0].values.shape[0] == 7


def test_multi_layer():
    model = IDiomTransformer(TINY).eval()
    out = extract_activations(model, _tokens(), layers=[0, 2])
    assert set(out) == {0, 2}
    assert out[0].values.shape == out[2].values.shape == (4, TINY.d_model)
