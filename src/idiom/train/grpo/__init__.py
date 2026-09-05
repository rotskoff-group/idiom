"""GRPO post-training.

Modules:
    core: the GRPO objective as pure tensor functions.
    data: prompt datasets and their collate function.
    lit_grpo: the LightningModule running rollout, reward, and the GRPO step.
    train_grpo: the idiom_train_grpo entrypoint.
    reward: the reward and shaping specs and the total-reward composition.
"""

from idiom.train.grpo.core import group_advantages, grpo_loss, sequence_kl, sequence_logprobs
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.reward import REWARD_ALIASES, SHAPING_ALIASES, build_reward, lift

__all__ = [
    "REWARD_ALIASES",
    "SHAPING_ALIASES",
    "LitGRPO",
    "build_reward",
    "grpo_loss",
    "group_advantages",
    "lift",
    "sequence_kl",
    "sequence_logprobs",
]
