"""GRPO reward subsystem.

Modules:
    base: the reward registry and the entropy and length terms.
    external_reward: subprocess scorers that run a reward model in its own environment.
    rl_sae_reward: rewards for reproducing a target's SAE feature code, imported on demand.
    compose_reward: the weighted-sum composition that LitGRPO calls.

User-editable reward content — copyable in-process rewards, external scorer programs, and SAE
signature files — lives in the repository's top-level rewards/ directory.
"""

from idiom.train.grpo.reward.base import (
    REWARD_REGISTRY,
    entropy_reward,
    get_reward,
    length_reward,
    quadratic_penalty,
    register_reward,
    resolve_reward,
    sequence_entropy,
)
from idiom.train.grpo.reward.compose_reward import build_reward_terms
from idiom.train.grpo.reward.external_reward import (
    Scorer,
    make_external_reward,
    parse_response,
    target_penalty,
)

__all__ = [
    "REWARD_REGISTRY",
    "Scorer",
    "build_reward_terms",
    "entropy_reward",
    "get_reward",
    "length_reward",
    "make_external_reward",
    "parse_response",
    "quadratic_penalty",
    "register_reward",
    "resolve_reward",
    "sequence_entropy",
    "target_penalty",
]
