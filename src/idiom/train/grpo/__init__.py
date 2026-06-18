"""GRPO post-training (P4).

Clean reimplementation of the legacy ``scores.py`` + ``grpo_loss.py``: rewards operate on the
**decoded IDR string** (``f(idr) -> float``), and the GRPO objective is a few pure tensor
functions (``train/grpo/core.py``). Online generation uses the KV-cached sampler. Heavy rewards
(ProtGPS) load a vendored model and are operator-wired; the fraction/length/entropy rewards are
pure and tested.
"""

from idiom.train.grpo.core import grpo_loss, group_advantages, sequence_kl, sequence_logprobs
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.rewards import REWARD_REGISTRY, get_reward, register_reward

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
