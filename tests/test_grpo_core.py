"""P4 GRPO-core tests (CPU-only): rewards + advantages + loss + logprobs."""

import math

import torch

from idiom.model import IDiomTransformer, ModelConfig
from idiom.train.grpo import grpo_loss, group_advantages, sequence_logprobs
from idiom.train.grpo.rewards import (
    entropy_reward,
    fraction_proline,
    get_reward,
    length_reward,
    sequence_entropy,
)


def test_rewards():
    assert fraction_proline("PPAP") == 0.75
    assert fraction_proline("") == 0.0
    assert get_reward("fraction_alanine")("AAAA") == 1.0
    # entropy is in bits (log2): single residue -> 0; uniform over k -> log2(k)
    assert sequence_entropy("AAAA") == 0.0
    assert abs(sequence_entropy("ACDE") - math.log2(4)) < 1e-9
    # quadratic penalties (legacy): 0 at the target, negative away from it
    assert abs(length_reward("A" * 100, target_length=100)) < 1e-9
    assert length_reward("A" * 50, target_length=100) < 0.0
    assert entropy_reward("ACDE", target_entropy=math.log2(4)) > entropy_reward("AAAA", target_entropy=math.log2(4))


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
