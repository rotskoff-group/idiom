"""P2 model tests (CPU-only): RoPE, RMSNorm, forward shapes, and KV-cache equivalence."""

import torch

from idiom.model import IDiomTransformer, KVCache, ModelConfig
from idiom.model.norms import RMSNorm
from idiom.model.rope import Rope

TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=16)


def _tiny_model():
    return IDiomTransformer(TINY).eval()


def test_rmsnorm_unit_scale():
    norm = RMSNorm(8)
    x = torch.randn(4, 8) * 5.0
    out = norm(x)  # weight initialized to ones -> output rows have RMS ~1
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-3)


def test_rope_is_norm_preserving():
    rope = Rope(head_dim=8, max_seq_len=16)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    pos = torch.arange(5)
    qr, kr = rope.apply(q, k, pos)
    # rotation preserves per-vector norm
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5)
    assert qr.shape == q.shape and kr.shape == k.shape


def test_forward_shapes_and_tied_embeddings():
    model = _tiny_model()
    tokens = torch.randint(0, TINY.vocab_size, (2, 6))
    logits = model(tokens)
    assert logits.shape == (2, 6, TINY.vocab_size)
    # tied embeddings: lm_head shares the embedding weight matrix
    assert model.lm_head.weight is model.embed.weight
    logits2, hidden = model(tokens, return_hidden_states=True)
    assert len(hidden) == TINY.n_layers and hidden[0].shape == (2, 6, TINY.d_model)


@torch.no_grad()
def test_kv_cache_matches_full_forward():
    model = _tiny_model()
    tokens = torch.randint(0, TINY.vocab_size, (1, 7))

    full = model(tokens)  # [1, 7, V]

    # Decode one token at a time through a KV cache; should reproduce the full logits.
    cache = KVCache(TINY.n_layers)
    steps = [model(tokens[:, t : t + 1], cache=cache) for t in range(tokens.size(1))]
    incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-4)
