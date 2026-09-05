"""Trivial reward factories used only by the test suite.

Config-driven tests name these by their "module:function" path, e.g.
reward: tests.reward_fixtures:fraction_proline.
"""

from omegaconf import OmegaConf

from idiom.train.grpo.reward import Reward, build_reward, lift


def fraction_proline() -> Reward:
    """Build a reward scoring the fraction of residues that are proline."""
    return lift(lambda idr: idr.count("P") / len(idr) if idr else 0.0)


def fraction_alanine() -> Reward:
    """Build a reward scoring the fraction of residues that are alanine."""
    return lift(lambda idr: idr.count("A") / len(idr) if idr else 0.0)


def scaled(residue: str = "P", scale: float = 1.0) -> Reward:
    """Build a reward counting one residue and scaling it -- a factory that takes arguments."""
    return lift(lambda idr: scale * idr.count(residue))


def proline_terms():
    """Build a composite reward over fraction_proline alone, for LitGRPO's reward_terms."""
    return build_reward(OmegaConf.create(
        {"terms": [{"reward": "tests.reward_fixtures:fraction_proline", "weight": 1.0}]}))
