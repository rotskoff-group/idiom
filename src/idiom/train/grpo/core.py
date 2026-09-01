"""GRPO objective as pure tensor functions (DAPO-style).

sequence_logprobs gives per-token log-probs of a sequence under a model. group_advantages turns
rewards into advantages normalized within each prompt's group. grpo_loss is the PPO-clipped
policy gradient minus a Schulman KL penalty to a reference, aggregated token-level over the whole
batch (DAPO).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprobs(model, tokens: Tensor) -> Tensor:
    """Return per-token log-probs of tokens[:, 1:] under the model.

    Args:
        model: Model mapping token ids to next-token logits.
        tokens (Tensor): Token id sequence of shape [B, L].

    Returns:
        Tensor: Per-token log-probs of shape [B, L-1].
    """
    logits = model(tokens)[:, :-1]  # predict position t+1 from <=t
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)


def _kl_per_token(policy_logp: Tensor, ref_logp: Tensor) -> Tensor:
    """Schulman k3 per-token KL estimate of policy to reference: exp(d) - d - 1 with d = ref - policy."""
    d = ref_logp - policy_logp
    return torch.exp(d) - d - 1.0


def sequence_kl(policy_logp: Tensor, ref_logp: Tensor, completion_mask: Tensor) -> Tensor:
    """Return the masked token-level mean of the Schulman KL, the same penalty grpo_loss applies.

    Args:
        policy_logp (Tensor): Per-token log-probs under the policy.
        ref_logp (Tensor): Per-token log-probs under the reference.
        completion_mask (Tensor): Mask selecting completion tokens.

    Returns:
        Tensor: Scalar mean KL over the masked tokens (for logging).
    """
    kl = _kl_per_token(policy_logp, ref_logp)
    return (kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)


def group_advantages(
    rewards: Tensor, group_size: int, *, normalize: bool = True, eps: float = 1e-8
) -> Tensor:
    """Compute advantages as reward minus group mean, optionally divided by the group std.

    Rewards are reshaped into groups of group_size (one group per prompt).

    Args:
        rewards (Tensor): Flat rewards of shape [B*G].
        group_size (int): Number of completions per prompt group.
        normalize (bool): If True, divide each advantage by its group's standard deviation.
        eps (float): Floor on the group std to avoid division by zero.

    Returns:
        Tensor: Flat advantages of shape [B*G].
    """
    grouped = rewards.view(-1, group_size)
    adv = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize:
        std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)
        adv = adv / std
    return adv.reshape(-1)


def grpo_loss(
    policy_logp: Tensor,
    ref_logp: Tensor,
    advantages: Tensor,
    completion_mask: Tensor,
    *,
    beta_kl: float = 0.0,
    eps_clip: float = 0.2,
) -> Tensor:
    """Compute the DAPO GRPO loss over completion tokens.

    The ratio exp(logp - logp.detach()) is identically 1 but carries the policy gradient (the TRL
    trick); advantages are already folded in. The KL term is the Schulman approximation to the
    reference. Aggregation is the sum over masked tokens divided by the total masked tokens (a
    token-level mean).

    Args:
        policy_logp (Tensor): Policy per-token log-probs of shape [B, T].
        ref_logp (Tensor): Reference per-token log-probs of shape [B, T].
        advantages (Tensor): Per-sequence advantages of shape [B].
        completion_mask (Tensor): Completion-token mask of shape [B, T].
        beta_kl (float): Weight of the KL penalty (0 disables it).
        eps_clip (float): PPO clipping range around a ratio of 1.

    Returns:
        Tensor: The scalar GRPO loss.
    """
    adv = advantages[:, None]
    ratio = torch.exp(policy_logp - policy_logp.detach())
    clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg = torch.min(ratio * adv, clipped * adv)

    if beta_kl > 0:
        per_token = -(pg - beta_kl * _kl_per_token(policy_logp, ref_logp))
    else:
        per_token = -pg

    per_token = per_token * completion_mask
    return per_token.sum() / completion_mask.sum().clamp(min=1.0)
