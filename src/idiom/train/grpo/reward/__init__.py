"""GRPO reward subsystem.

A reward term is a reward, a shaping rule, and a weight. The total reward is the weighted sum of
the shaped rewards over the configured terms. Nothing is in an objective unless a run named it:
reward.terms is empty by default and the library ships no presets.

A term is four keys, and the reward and the shaping are named the same way -- a factory plus its
arguments, as a bare name or a mapping:

    {reward: entropy, shaping: {name: quadratic, target: 3.65, width: 0.2}, weight: 1.0}
    {reward: {name: scorer, cmd: "uv run --script my_scorer.py"}, label: mine, weight: 1.0}
    {reward: {name: "mypkg.scoring:make_scorer", cutoff: 0.3}, label: mine, weight: 1.0}

Modules:
    resolve: what a reward is, the shipped aliases, and how a spec becomes a callable.
    builtin: the entropy and length reward factories.
    shaping: the shipped shaping rules, and the arithmetic behind them.
    external: scorer, the reward factory that runs a reward model in its own environment.
    compose: config validation and the weighted-sum composition that LitGRPO calls.

Worked external scorers live in the repository at cookbook/rewards/scorers/. The SAE feature
reward, sae_feature.sae_signature, is imported on demand, since it pulls in the SAE.
"""

from idiom.train.grpo.reward.builtin import composition_entropy, entropy, length
from idiom.train.grpo.reward.compose import TERM_KEYS, Term, build_reward, build_terms
from idiom.train.grpo.reward.external import Scorer, parse_response, scorer
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
    "Scorer",
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
