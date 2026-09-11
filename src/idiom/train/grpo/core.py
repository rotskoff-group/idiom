"""Token-level GRPO loss, group advantages, and reference-policy KL estimates."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprobs(model, tokens: Tensor) -> Tensor:
    """Return log-probabilities of tokens[:, 1:] given their prefixes.

    Input token ids have shape [B, L]; output has shape [B, L-1].
    """
    logits = model(tokens[:, :-1]) # the final target need not fit as an input position
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)


def _kl_per_token(policy_logp: Tensor, ref_logp: Tensor) -> Tensor:
    """Return the Schulman k3 per-token KL estimate, exp(d) - d - 1 with d = ref - policy."""
    d = ref_logp - policy_logp
    return torch.exp(d) - d - 1.0


def sequence_kl(policy_logp: Tensor, ref_logp: Tensor, completion_mask: Tensor) -> Tensor:
    """Return mean Schulman k3 KL over completion_mask; return 0 for an empty mask.

    All inputs have shape [B, T]. The estimate is exp(d) - d - 1, where d = ref - policy.
    """
    kl = _kl_per_token(policy_logp, ref_logp)
    return (kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)


def group_advantages(
    rewards: Tensor, group_size: int, *, normalize: bool = True, eps: float = 1e-8
) -> Tensor:
    """Compute advantages as each reward minus its group mean.

    Rewards are reshaped into consecutive groups of group_size, one group per prompt.

    Args:
        rewards: Flat rewards of shape [B*G].
        group_size: Number of completions per prompt group.
        normalize: If True, also divide each advantage by its group's standard deviation.
        eps: Floor on the group standard deviation.

    Returns:
        Flat advantages of shape [B*G].
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
    """Return token-mean negative clipped policy gain plus beta_kl * KL.

    The importance ratio uses detached current-policy log-probabilities as its baseline.

    Args:
        policy_logp: Policy per-token log-probs of shape [B, T].
        ref_logp: Reference per-token log-probs of shape [B, T].
        advantages: Per-sequence advantages of shape [B].
        completion_mask: Completion-token mask of shape [B, T].
        beta_kl: Weight of the KL penalty; 0 omits the term.
        eps_clip: PPO clipping range around a ratio of 1.

    Returns:
        The scalar GRPO loss.
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
