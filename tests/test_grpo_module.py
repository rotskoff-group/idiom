"""P4 GRPO module test (CPU-only): a full generate->reward->loss step runs and backprops."""

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.model import ModelConfig
from idiom.train.grpo import LitGRPO
from idiom.train.grpo.rewards import fraction_proline

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=64)


def test_grpo_step_runs_and_backprops():
    lit = LitGRPO(TINY, fraction_proline, group_size=2, max_new_tokens=6, beta_kl=0.02)
    prompts = torch.tensor([TOK.encode("132")])  # one de-novo prompt -> group of 2 completions
    loss = lit.training_step(prompts, 0)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    # reference stays frozen; policy receives gradients
    assert all(not p.requires_grad for p in lit.reference.parameters())
    assert any(p.grad is not None for p in lit.model.parameters())


def test_reference_starts_equal_to_policy():
    lit = LitGRPO(TINY, fraction_proline, group_size=2)
    for (_, a), (_, b) in zip(lit.model.state_dict().items(), lit.reference.state_dict().items()):
        assert torch.equal(a, b)
