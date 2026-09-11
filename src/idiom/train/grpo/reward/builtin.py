"""Composition entropy and residue-count rewards."""

from __future__ import annotations

import math
from collections import Counter

from idiom.train.grpo.reward.resolve import Reward, batchify


def composition_entropy(idr: str) -> float:
    """Return amino-acid composition entropy in bits; empty strings score 0.

    For canonical sequences the range is [0, log2(20)].
    """
    if not idr:
        return 0.0
    n = len(idr)
    h = -sum((c / n) * math.log2(c / n) for c in Counter(idr).values())
    return h or 0.0 # a single repeated residue gives -0.0; log it as 0.0


def entropy() -> Reward:
    """Build a composition-entropy reward in bits."""
    return batchify(composition_entropy)


def length() -> Reward:
    """Build a sequence-length reward in residues."""
    return batchify(len)
