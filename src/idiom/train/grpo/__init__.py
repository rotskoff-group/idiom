"""GRPO training, prompt datasets, and configurable rewards."""

from idiom.train.grpo.core import group_advantages, grpo_loss, sequence_kl, sequence_logprobs
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.reward import REWARD_ALIASES, SHAPING_ALIASES, batchify, build_reward

__all__ = [
    "REWARD_ALIASES",
    "SHAPING_ALIASES",
    "LitGRPO",
    "build_reward",
    "grpo_loss",
    "group_advantages",
    "batchify",
    "sequence_kl",
    "sequence_logprobs",
]
