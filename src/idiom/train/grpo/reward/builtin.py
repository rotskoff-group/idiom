"""The two guardrail rewards, and the only ones the library itself defines.

entropy and length are here because every GRPO run should carry them: they are what stops an
objective from being satisfied by a low-complexity tract or by drifting to a degenerate length.
Being library code, they work in any install -- a non-editable `pip install git+...` included --
so the shipped config names them with no module and nothing on disk beside it.

Nothing else is built in. A reward that expresses what *you* want to design is yours to write, and
the library should not pretend to have opinions about which ones matter: see
cookbook/rewards/custom_rewards.py, which carries worked examples (charge, motif density) as a template
to copy, and cookbook/rewards/scorers/ for reward models that run in their own environment.

Importing idiom.train.grpo.reward registers both, so a term names one directly:

    reward.terms:
      - {reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}
      - {reward: length,  weight: 1.0, shaping: {type: quadratic, target: 100,  width: 1.0}}

A reward reports one raw value in whatever units suit it and says nothing about what a good value
is -- that is the term's shaping -- so the same function serves as a target, a guardrail, or a
logged-only diagnostic (weight 0) without being rewritten.
"""

from __future__ import annotations

import math
from collections import Counter

from idiom.train.grpo.reward.registry import register_reward


@register_reward("entropy")
def sequence_entropy(idr: str) -> float:
    """Return the Shannon entropy of an IDR's amino-acid composition, in bits.

    The value ranges from 0 for a single repeated residue to log2(20), about 4.32 bits, for a
    uniform composition. Natural IDRs sit near 3.65 bits. One half of the shipped guardrail pair:
    it is what stops an objective from being satisfied by a low-complexity tract.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Composition entropy in bits, or 0.0 for an empty string.
    """
    if not idr:
        return 0.0
    n = len(idr)
    h = -sum((c / n) * math.log2(c / n) for c in Counter(idr).values())
    return h or 0.0  # a single repeated residue gives -0.0; log it as 0.0


@register_reward("length")
def sequence_length(idr: str) -> float:
    """Return an IDR's length in residues.

    The other half of the shipped guardrail pair: without it, a reward with no length preference is
    free to drift to the shortest or longest sequence that satisfies it.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Number of residues.
    """
    return float(len(idr))
