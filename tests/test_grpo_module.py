"""GRPO module test: a full generate->reward->loss step runs and backprops."""

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.model import ModelConfig
from idiom.train.grpo import LitGRPO
from tests.reward_fixtures import proline_terms

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=64)


def test_grpo_step_runs_and_backprops():
    lit = LitGRPO(TINY, proline_terms(), group_size=2, max_new_tokens=6, beta_kl=0.02)
    prompts = torch.tensor([TOK.encode("132")])
    loss = lit.training_step(prompts, 0)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert all(not p.requires_grad for p in lit.reference.parameters())
    assert any(p.grad is not None for p in lit.model.parameters())


def test_reference_starts_equal_to_policy():
    lit = LitGRPO(TINY, proline_terms(), group_size=2)
    for (_, a), (_, b) in zip(lit.model.state_dict().items(), lit.reference.state_dict().items()):
        assert torch.equal(a, b)


def test_context_clamped_grpo_rollout_can_be_rescored():
    import pytest

    cfg = ModelConfig(n_layers=1, d_model=16, n_heads=2, max_seq_len=8)
    lit = LitGRPO(cfg, proline_terms(), group_size=2, max_new_tokens=100, temperature=0, log_samples_every=0)
    # Equal logits force greedy residue 0, so no STOP shortens the boundary-length rollout
    with torch.no_grad():
        lit.model.lm_head.weight.zero_()
    with pytest.warns(UserWarning, match="remaining context"):
        loss = lit.training_step(torch.tensor([TOK.encode("132")]), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in lit.model.parameters() if p.grad is not None)
