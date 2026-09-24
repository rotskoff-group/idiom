"""Importable reward factories for tests."""

from omegaconf import OmegaConf

from idiom.train.grpo.reward import Reward, batchify, build_reward


def fraction_proline() -> Reward:
    """Build a reward scoring the fraction of residues that are proline."""
    return batchify(lambda idr: idr.count("P") / len(idr) if idr else 0.0)


def fraction_alanine() -> Reward:
    """Build a reward scoring the fraction of residues that are alanine."""
    return batchify(lambda idr: idr.count("A") / len(idr) if idr else 0.0)


def scaled(residue: str = "P", scale: float = 1.0) -> Reward:
    """Build a scaled residue-count reward."""
    return batchify(lambda idr: scale * idr.count(residue))


def proline_terms():
    """Build a composite reward over fraction_proline alone, for LitGRPO's reward_terms."""
    return build_reward(
        OmegaConf.create({"terms": [{"reward": "tests.reward_fixtures:fraction_proline", "weight": 1.0}]})
    )
