"""P2 sampling tests (CPU-only): KV-cached generation + sampling controls."""

import torch

from idiom.data.fim import fim_prompt
from idiom.data.tokenizer import Tokenizer
from idiom.model import IDiomTransformer, ModelConfig, generate

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=32)


def _model():
    return IDiomTransformer(TINY).eval()


def test_fim_prompt():
    assert fim_prompt() == "132"  # de-novo IDP
    assert fim_prompt("MEDSKVDNRPQ", 4, 8) == "1MEDS3RPQ2"  # flanks of IDR seq[4:8]


@torch.no_grad()
def _ref_greedy(model, prompt, n_new):
    seq = torch.cat([torch.full((prompt.size(0), 1), TOK.start_id), prompt], dim=1)
    out = []
    for _ in range(n_new):
        nxt = model(seq)[:, -1].argmax(-1)  # full forward, no cache
        out.append(nxt)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    return torch.stack(out, dim=1)


def test_greedy_matches_uncached_reference():
    model = _model()
    prompt = torch.tensor(TOK.encode("132")).unsqueeze(0)  # [1, 3]
    # greedy via KV cache must equal greedy via repeated full forwards (no cache).
    cached = generate(model, prompt, max_new_tokens=8, temperature=0, stop_id=None)
    ref = _ref_greedy(model, prompt, 8)
    assert torch.equal(cached, ref)


def test_greedy_is_deterministic():
    model = _model()
    prompt = torch.tensor(TOK.encode("132")).unsqueeze(0)
    a = generate(model, prompt, max_new_tokens=6, temperature=0, stop_id=None)
    b = generate(model, prompt, max_new_tokens=6, temperature=0, stop_id=None)
    assert torch.equal(a, b)


def test_sampling_in_range_and_seeded():
    model = _model()
    prompt = torch.tensor([TOK.encode("132"), TOK.encode("132")])  # batch of 2
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=5, top_p=0.9, generator=g1)
    out2 = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=5, top_p=0.9, generator=g2)
    assert out1.shape[0] == 2 and out1.size(1) <= 10
    assert torch.equal(out1, out2)  # same seed -> same samples
    assert int(out1.max()) < TINY.vocab_size and int(out1.min()) >= 0
