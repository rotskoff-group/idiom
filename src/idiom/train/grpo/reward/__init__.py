"""GRPO reward subsystem.

A reward term is a reward and its shaping: the reward reports a raw value in natural units, shaping
says what a good value is, and the term's weight sets how much it matters. The total reward is the
weighted sum of the shaped rewards over the configured terms.

Modules:
    registry: the reward registry and the per-batch context rewards receive.
    builtin: the pure-python rewards that ship, registered by importing this package.
    shaping: the shaping rules a term can apply to a raw reward, and their registry.
    external: subprocess scorers that run a reward model in its own environment.
    compose: config validation and the weighted-sum composition that LitGRPO calls.

A reward reaches a run one of three ways, and they differ only in where the raw value comes from:

- builtin -- entropy and length, the guardrails, and nothing else; registered here, so the
  shipped config names them with no module and no clone;
- yours, in-process -- a term names an importable module (which registers it) or a callable
  directly as "package.module:function", so code already installed alongside IDiom is used as is;
- yours, out-of-process -- a term names a cmd, and external.py speaks JSON to it over a pipe, for a
  reward model whose dependencies cannot coexist with IDiom's.

Only the first two are library code. A scorer for the third is a standalone program with its own
dependencies, so none ship inside the package: the worked ones live in the repository at
cookbook/rewards/scorers/ and a term names one with an explicit cmd. The SAE feature reward is an
ordinary module, sae_feature, imported on demand because it pulls in the SAE.
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
