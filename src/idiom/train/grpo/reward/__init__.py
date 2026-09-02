"""GRPO reward subsystem.

Four pieces: base (the registry and the entropy/length guardrail terms), external_reward
(subprocess scorers that run a reward model in its own environment), rl_sae_reward (reward for
reproducing a target's SAE feature code), and compose_reward (the weighted-sum composition that
LitGRPO actually calls).

The split by directory is deliberate: this package holds reward *machinery*, while the repo's
top-level rewards/ directory holds user-editable reward *content* — example_rewards.py to copy,
external_scorers/ to run in foreign environments, and rl_sae_targets/ signature files. rl_sae_reward
lives here rather than there only because it imports the idiom SAE; it is imported on demand, when
the rl_sae term is enabled, so nothing pays for loading a model it is not using.
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
