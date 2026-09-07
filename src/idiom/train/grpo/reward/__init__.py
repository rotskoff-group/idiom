"""Configurable GRPO rewards: raw batch scores, scalar shaping, and weighted composition.

Terms name factories by alias or "module:function" path. See cookbook/rewards for examples.
"""

from idiom.train.grpo.reward.builtin import composition_entropy, entropy, length
from idiom.train.grpo.reward.compose import TERM_KEYS, Term, build_reward, build_terms
from idiom.train.grpo.reward.external import ScorerProcess, parse_response, scorer
from idiom.train.grpo.reward.resolve import (
    REWARD_ALIASES,
    SHAPING_ALIASES,
    Reward,
    build_from_spec,
    import_module,
    lift,
    load_callable,
    spec_name,
)
from idiom.train.grpo.reward.shaping import (
    Shaping,
    gaussian,
    gaussian_score,
    identity,
    quadratic,
    quadratic_penalty,
    tolerance,
)

__all__ = [
    "REWARD_ALIASES",
    "SHAPING_ALIASES",
    "Reward",
    "ScorerProcess",
    "Shaping",
    "TERM_KEYS",
    "Term",
    "build_from_spec",
    "build_reward",
    "build_terms",
    "composition_entropy",
    "entropy",
    "gaussian",
    "gaussian_score",
    "identity",
    "import_module",
    "length",
    "lift",
    "load_callable",
    "parse_response",
    "quadratic",
    "quadratic_penalty",
    "scorer",
    "spec_name",
    "tolerance",
]
