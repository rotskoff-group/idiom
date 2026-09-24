"""Sampling tests: KV-cached generation + sampling controls."""

import torch

from idiom.data.fim import fim_prompt
from idiom.data.tokenizer import Tokenizer
from idiom.model import IDiomTransformer, ModelConfig, generate

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=32)


def _model():
    """Create a tiny transformer in evaluation mode for sampling tests."""
    return IDiomTransformer(TINY).eval()


def test_fim_prompt():
    """Verify the bare and flank-conditioned FIM generation prompts."""
    assert fim_prompt() == "132"
    assert fim_prompt("MEDSKVDNRPQ", 4, 8) == "1MEDS3RPQ2"


@torch.no_grad()
def _ref_greedy(model, prompt, n_new):
    """Generate greedy tokens using full-prefix forward passes without a KV cache."""
    seq = torch.cat([torch.full((prompt.size(0), 1), TOK.start_id), prompt], dim=1)
    out = []
    for _ in range(n_new):
        nxt = model(seq)[:, -1].argmax(-1)  # full forward, no cache
        out.append(nxt)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    return torch.stack(out, dim=1)


def test_greedy_matches_uncached_reference():
    """Verify that cached greedy decoding matches an uncached reference."""
    model = _model()
    prompt = torch.tensor(TOK.encode("132")).unsqueeze(0)
    cached = generate(model, prompt, max_new_tokens=8, temperature=0, stop_id=None)
    ref = _ref_greedy(model, prompt, 8)
    assert torch.equal(cached, ref)


def test_greedy_is_deterministic():
    """Verify repeatable greedy generation."""
    model = _model()
    prompt = torch.tensor(TOK.encode("132")).unsqueeze(0)
    a = generate(model, prompt, max_new_tokens=6, temperature=0, stop_id=None)
    b = generate(model, prompt, max_new_tokens=6, temperature=0, stop_id=None)
    assert torch.equal(a, b)


def test_sampling_in_range_and_seeded():
    """Verify valid token IDs and repeatability with a fixed sampling seed."""
    model = _model()
    prompt = torch.tensor([TOK.encode("132"), TOK.encode("132")])
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=5, top_p=0.9, generator=g1)
    out2 = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=5, top_p=0.9, generator=g2)
    assert out1.shape[0] == 2 and out1.size(1) <= 10
    assert torch.equal(out1, out2)
    assert int(out1.max()) < TINY.vocab_size and int(out1.min()) >= 0


def test_context_boundary_matches_reference_without_extra_forward():
    """Verify context-limited sampling without an unnecessary final forward pass."""
    import pytest

    model = _model()
    prompt = torch.tensor([TOK.encode("132")])
    available = TINY.max_seq_len - prompt.size(1)  # includes the final next-token prediction
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
    with pytest.warns(UserWarning, match="remaining context"):
        out = generate(model, prompt, max_new_tokens=1000, temperature=0, stop_id=None)
    handle.remove()
    assert out.shape == (1, available)
    assert len(calls) == available
    assert torch.equal(out, _ref_greedy(model, prompt, available))


def test_oversized_prompt_rejected_before_forward():
    """Verify that oversized prompts fail before model execution."""
    import pytest

    model = _model()
    prompt = torch.zeros((1, TINY.max_seq_len), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeding context length"):
        generate(model, prompt, max_new_tokens=1)
