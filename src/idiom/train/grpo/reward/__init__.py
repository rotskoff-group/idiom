"""GRPO reward subsystem.

A reward term is a reward and its shaping: the reward reports a raw value in natural units, shaping
says what a good value is, and the term's weight sets how much it matters. The total reward is the
weighted sum of the shaped rewards over the configured terms.

Modules:
    registry: the reward registry and the per-batch context rewards receive.
    shaping: the shaping rules a term can apply to a raw reward, and their registry.
    external: subprocess scorers that run a reward model in its own environment.
    compose: config validation and the weighted-sum composition that LitGRPO calls.

The rewards themselves are not here: every one that ships -- the entropy and length guardrails, the
charge and motif examples, the SAE feature reward, and the external scorer programs -- lives in the
repository's top-level rewards/ directory, as material you edit rather than library code. This
package is the machinery that registers, shapes, and composes them, and it imports none of it.
"""

from idiom.train.grpo.reward.compose import (
    RewardTermSpec,
    build_reward,
    import_module_spec,
    parse_terms,
)
from idiom.train.grpo.reward.external import (
    Scorer,
    make_external_reward,
    parse_response,
)
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
    "RewardTermSpec",
    "SHAPING_REGISTRY",
    "Batch",
    "Scorer",
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
    "tolerance",
]
