"""GRPO-core tests: rewards + advantages + loss + logprobs."""

import math

import torch

from idiom.model import IDiomTransformer, ModelConfig
from idiom.train.grpo import group_advantages, grpo_loss, sequence_logprobs
from idiom.train.grpo.reward import entropy, length, quadratic_penalty
from tests.reward_fixtures import fraction_alanine, fraction_proline


def test_rewards():
    assert fraction_proline()(["PPAP", ""]) == [0.75, 0.0]
    assert fraction_alanine()(["AAAA", "AC"]) == [1.0, 0.5]
    assert entropy()(["AAAA"]) == [0.0]
    assert abs(entropy()(["ACDE"])[0] - math.log2(4)) < 1e-9
    assert length()(["A" * 100]) == [100.0]
    assert abs(quadratic_penalty(100.0, 100.0)) < 1e-9
    assert quadratic_penalty(50.0, 100.0) < 0.0


def test_group_advantages():
    rewards = torch.tensor([1.0, 3.0, 0.0, 2.0])  # two groups of 2
    adv = group_advantages(rewards, group_size=2, normalize=False)
    assert adv.tolist() == [-1.0, 1.0, -1.0, 1.0]
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
    ref = torch.log_softmax(model(tokens)[:, :-1].float(), -1)
    expected = ref.gather(-1, tokens[:, 1:, None]).squeeze(-1)
    assert torch.allclose(logp, expected, atol=1e-6)


def test_sequence_logprobs_at_context_boundary_matches_prefix_scoring():
    cfg = ModelConfig(n_layers=1, d_model=16, n_heads=2, max_seq_len=8)
    model = IDiomTransformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len + 1))
    actual = sequence_logprobs(model, tokens)
    expected = torch.stack(
        [
            model(tokens[:, :i]).float().log_softmax(-1)[:, -1].gather(1, tokens[:, i : i + 1]).squeeze(1)
            for i in range(1, tokens.size(1))
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    (-actual.mean()).backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()
