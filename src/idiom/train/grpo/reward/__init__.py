"""GRPO reward subsystem.

A reward term is a raw reward, a shaping rule, and a weight. The total reward is the weighted sum
of the shaped rewards over the configured terms.

Modules:
    registry: the reward registry and the per-batch context rewards receive.
    builtin: the entropy and length rewards, registered by importing this package.
    shaping: the shaping rules a term can apply to a raw reward, and their registry.
    external: subprocess scorers that run a reward model in its own environment.
    compose: config validation and the weighted-sum composition that LitGRPO calls.

A term names its raw reward in one of three ways:

- a registered name, such as the built-in entropy and length;
- "package.module:function", any importable callable, with no decorator;
- a cmd, an external scorer spoken to in JSON over a pipe (see external).

Worked external scorers live in the repository at cookbook/rewards/scorers/. The SAE feature
reward, sae_feature, is imported on demand, since it pulls in the SAE.
"""

from idiom.train.grpo.reward import builtin  # noqa: F401 - registers the shipped rewards
from idiom.train.grpo.reward.builtin import sequence_entropy, sequence_length
from idiom.train.grpo.reward.compose import (
    RewardTermSpec,
    build_reward,
    import_module_spec,
    is_callable_spec,
    load_callable,
    parse_terms,
    resolve_add,
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
    "is_callable_spec",
    "load_callable",
    "make_external_reward",
    "parse_response",
    "parse_terms",
    "quadratic_penalty",
    "register_reward",
    "register_shaping",
    "resolve_add",
    "sequence_entropy",
    "sequence_length",
    "tolerance",
]
