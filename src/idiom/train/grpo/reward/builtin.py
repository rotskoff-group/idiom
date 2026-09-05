"""Two sequence-level rewards the library ships, entropy and length.

Neither is in an objective unless a run names it: reward.terms is empty by default and both are
terms like any other. Most objectives are worth carrying them, since a target is otherwise
satisfiable by a low-complexity tract or by a degenerate length.

    reward.terms:
      - {reward: entropy, shaping: {name: quadratic, target: 3.65, width: 0.2}, weight: 1.0}
      - {reward: length,  shaping: {name: quadratic, target: 100,  width: 1.0}, weight: 1.0}

Write your own rewards and shaping from the template in cookbook/rewards/custom_rewards.py, or as
external scorers in cookbook/rewards/scorers/.
"""

from __future__ import annotations

import math
from collections import Counter

from idiom.train.grpo.reward.resolve import Reward, lift


def composition_entropy(idr: str) -> float:
    """Return the Shannon entropy of an IDR's amino-acid composition, in bits.

    The value ranges from 0 for a single repeated residue to log2(20), about 4.32 bits, for a
    uniform composition. Natural IDRs sit near 3.65 bits.

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


def entropy() -> Reward:
    """Build the reward scoring an IDR's composition entropy in bits.

    Returns:
        Reward: Composition entropy per IDR; see composition_entropy.
    """
    return lift(composition_entropy)


def length() -> Reward:
    """Build the reward scoring an IDR's length in residues.

    Returns:
        Reward: Number of residues per IDR.
    """
    return lift(len)
