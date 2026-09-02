"""GRPO reward subsystem.

Three pieces: base (the registry and the entropy/length terms), external_reward (subprocess scorers
that run a reward model in its own environment), and compose_reward (the weighted-sum composition).
The top-level rewards/ directory holds user content instead: example_rewards.py to copy, external_scorers/ run
in foreign environments, and rl_sae_targets/ signature files. The RL-SAE reward itself lives here
(reward/rl_sae_reward.py) because it imports the idiom SAE; it is imported on demand when enabled.
"""

from idiom.train.grpo.reward.base import (
    REWARD_REGISTRY,
    entropy_reward,
    get_reward,
    length_reward,
    quadratic_penalty,
    quadratic_shaping,
    register_reward,
    resolve_reward,
    sequence_entropy,
)
from idiom.train.grpo.reward.compose_reward import (
    build_reward,
    build_reward_components,
    build_reward_terms,
)
from idiom.train.grpo.reward.external_reward import (
    Scorer,
    make_external_reward,
    parse_response,
    target_penalty,
)

__all__ = [
    "REWARD_REGISTRY",
    "Scorer",
    "build_reward",
    "build_reward_components",
    "build_reward_terms",
    "entropy_reward",
    "get_reward",
    "length_reward",
    "make_external_reward",
    "parse_response",
    "quadratic_penalty",
    "quadratic_shaping",
    "register_reward",
    "resolve_reward",
    "sequence_entropy",
    "target_penalty",
]
