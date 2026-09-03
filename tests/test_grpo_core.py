"""P4 GRPO-core tests (CPU-only): rewards + advantages + loss + logprobs."""

import math

import torch

from idiom.model import IDiomTransformer, ModelConfig
from idiom.train.grpo import grpo_loss, group_advantages, sequence_logprobs
from idiom.train.grpo.reward import (
    Batch,
    get_reward,
    quadratic_penalty,
    sequence_entropy,
    sequence_length,
)
from reward_fixtures import fraction_proline


def test_rewards():
    assert fraction_proline("PPAP") == 0.75
    assert fraction_proline("") == 0.0
    # a registered reward is batched: one raw value per IDR, in order
    assert get_reward("fraction_alanine")(["AAAA", "AC"], Batch()) == [1.0, 0.5]
    # entropy is in bits (log2): single residue -> 0; uniform over k -> log2(k)
    assert sequence_entropy("AAAA") == 0.0
    assert abs(sequence_entropy("ACDE") - math.log2(4)) < 1e-9
    # rewards report raw units; nothing here knows about a target
    assert sequence_length("A" * 100) == 100.0
    # a target is the shaping's business: 0 at the target, negative away from it
    assert abs(quadratic_penalty(100.0, 100.0)) < 1e-9
    assert quadratic_penalty(50.0, 100.0) < 0.0


def test_group_advantages():
    rewards = torch.tensor([1.0, 3.0, 0.0, 2.0])  # two groups of 2
    adv = group_advantages(rewards, group_size=2, normalize=False)
    assert adv.tolist() == [-1.0, 1.0, -1.0, 1.0]  # reward - group mean
    norm = group_advantages(rewards, group_size=2, normalize=True)
    assert torch.allclose(norm.view(2, 2).mean(1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(norm.view(2, 2).std(1, unbiased=False), torch.ones(2), atol=1e-5)


def test_grpo_loss_grad_flows():
    B, T = 2, 4
    policy = torch.randn(B, T, requires_grad=True)
    ref = torch.randn(B, T)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(B, T)
    loss = grpo_loss(policy, ref, adv, mask, beta_kl=0.1, eps_clip=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()


def test_sequence_logprobs_shape_and_values():
    cfg = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=16)
    model = IDiomTransformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 5))
    logp = sequence_logprobs(model, tokens)
    assert logp.shape == (1, 4)
    # matches a manual log_softmax gather
    ref = torch.log_softmax(model(tokens)[:, :-1].float(), -1)
    expected = ref.gather(-1, tokens[:, 1:, None]).squeeze(-1)
    assert torch.allclose(logp, expected, atol=1e-6)
