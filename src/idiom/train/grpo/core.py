"""GRPO objective — pure tensor functions (DAPO-style, matching the legacy semantics).

- :func:`sequence_logprobs` — per-token log-probs of a sequence under a model.
- :func:`group_advantages` — reward → advantage, normalized within each prompt's group.
- :func:`grpo_loss` — PPO-clipped policy-gradient minus a Schulman-KL penalty to a reference,
  aggregated token-level over the whole batch (DAPO).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprobs(model, tokens: Tensor) -> Tensor:
    """Per-token log-prob of ``tokens[:, 1:]`` under ``model``. Shape ``[B, L-1]``."""
    logits = model(tokens)[:, :-1]  # predict position t+1 from <=t
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)


def _kl_per_token(policy_logp: Tensor, ref_logp: Tensor) -> Tensor:
    """Schulman k3 per-token KL estimate of policy→reference: ``exp(d) − d − 1`` with ``d = ref − policy``."""
    d = ref_logp - policy_logp
    return torch.exp(d) - d - 1.0


def sequence_kl(policy_logp: Tensor, ref_logp: Tensor, completion_mask: Tensor) -> Tensor:
    """Masked token-level mean of the Schulman KL — the same penalty :func:`grpo_loss` applies (for logging)."""
    kl = _kl_per_token(policy_logp, ref_logp)
    return (kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)


def group_advantages(
    rewards: Tensor, group_size: int, *, normalize: bool = True, eps: float = 1e-8
) -> Tensor:
    """Advantage = reward − group mean (÷ group std if ``normalize``). ``rewards`` is ``[B*G]``."""
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
    """DAPO GRPO loss over completion tokens. ``*_logp``/``mask`` are ``[B, T]``; ``advantages`` ``[B]``.

    The ratio ``exp(logp − logp.detach())`` is identically 1 but carries the policy gradient
    (TRL trick); advantages are already folded in. KL is the Schulman approximation to the
    reference. Aggregation is sum over masked tokens ÷ total masked tokens (token-level mean).
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
