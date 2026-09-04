"""GRPO post-training.

Modules:
    core: the GRPO objective as pure tensor functions.
    data: prompt datasets and their collate function.
    lit_grpo: the LightningModule running rollout, reward, and the GRPO step.
    train_grpo: the idiom_train_grpo entrypoint.
    reward: the reward and shaping registries and the total-reward composition.
"""

from idiom.train.grpo.core import group_advantages, grpo_loss, sequence_kl, sequence_logprobs
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
