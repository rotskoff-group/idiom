"""Trivial registered rewards used only by the test suite.

These are deliberately kept out of the shipped library (the real user-facing examples live in
rewards/example_rewards.py). conftest imports this module so the names resolve for config-driven
tests (e.g. reward.external name: fraction_proline); tests that call the functions directly import
them from here.
"""

from idiom.train.grpo.reward import register_reward


@register_reward("fraction_proline")
def fraction_proline(idr: str) -> float:
    """Fraction of residues in the IDR that are proline."""
    return idr.count("P") / len(idr) if idr else 0.0


@register_reward("fraction_alanine")
def fraction_alanine(idr: str) -> float:
    """Fraction of residues in the IDR that are alanine."""
    return idr.count("A") / len(idr) if idr else 0.0
