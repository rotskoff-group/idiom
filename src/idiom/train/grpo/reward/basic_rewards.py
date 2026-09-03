"""The built-in rewards: raw quantities read straight off an IDR string.

Each is a pure function of the residue string returning its natural unit -- entropy in bits, length
in residues -- unshaped, with no notion of a target. The config decides what a good value is (see
shaping), which is why a length target lives in configs/grpo.yaml rather than here.
"""

from __future__ import annotations

import math
from collections import Counter

from idiom.train.grpo.reward.registry import register_reward


@register_reward("entropy")
def sequence_entropy(idr: str) -> float:
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


@register_reward("length")
def sequence_length(idr: str) -> float:
    """Return an IDR's length in residues.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Number of residues.
    """
    return float(len(idr))
