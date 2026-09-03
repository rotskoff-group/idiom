"""GRPO reward subsystem.

A reward term is a reward and its shaping: the reward reports a raw value in natural units, shaping
says what a good value is, and the term's weight sets how much it matters. The total reward is the
weighted sum of the shaped rewards over the configured terms.

Modules:
    registry: the reward registry and the per-batch context rewards receive.
    rewards: the built-in rewards (composition entropy, length).
    shaping: the shaping rules a term can apply to a raw reward, and their registry.
    external: subprocess scorers that run a reward model in its own environment.
    rl_sae: rewards for reproducing a target's SAE feature code, imported on demand.
    compose: config validation and the weighted-sum composition that LitGRPO calls.

User-editable reward content -- copyable in-process rewards, external scorer programs, and SAE
signature files -- lives in the repository's top-level rewards/ directory.
"""

from idiom.train.grpo.reward.compose import (
    TermSpec,
    build_reward,
    import_module_spec,
    parse_terms,
)
from idiom.train.grpo.reward.external import (
    Scorer,
    make_external_reward,
    parse_response,
)
from idiom.train.grpo.reward.rewards import sequence_entropy, sequence_length
from idiom.train.grpo.reward.registry import (
    REWARD_REGISTRY,
    Batch,
    get_reward,
    register_reward,
)
from idiom.train.grpo.reward.shaping import (
    SHAPING_REGISTRY,
    build_shaping,
    gaussian_score,
    quadratic_penalty,
    register_shaping,
    tolerance,
)

__all__ = [
    "REWARD_REGISTRY",
    "SHAPING_REGISTRY",
    "Batch",
    "Scorer",
    "TermSpec",
    "build_reward",
    "build_shaping",
    "gaussian_score",
    "get_reward",
    "import_module_spec",
    "make_external_reward",
    "parse_response",
    "parse_terms",
    "quadratic_penalty",
    "register_reward",
    "register_shaping",
    "sequence_entropy",
    "sequence_length",
    "tolerance",
]
