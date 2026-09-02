"""GRPO post-training.

Rewards operate on the decoded IDR string (f(idr) -> float), and the GRPO objective is a few pure
tensor functions (train/grpo/core.py). Online generation uses the KV-cached sampler. Rewards that
need their own environment run as external subprocess scorers (train/grpo/reward/external_reward.py).
"""

from idiom.train.grpo.core import grpo_loss, group_advantages, sequence_kl, sequence_logprobs
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.reward import REWARD_REGISTRY, get_reward, register_reward

__all__ = [
    "REWARD_REGISTRY",
    "LitGRPO",
    "get_reward",
    "grpo_loss",
    "group_advantages",
    "register_reward",
    "sequence_kl",
    "sequence_logprobs",
]
