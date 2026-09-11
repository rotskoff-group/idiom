"""Templates for in-process GRPO rewards and shaping; copy and adapt.

Factories receive config arguments and return batch rewards or scalar shaping functions.
Use batchify for single-IDR scorers and "module:function" paths to name custom factories.
Validate settings in the factory. See cookbook/rewards/README.md for configuration.
"""

import re

from idiom.train.grpo.reward import Reward, Shaping, batchify


def net_charge_fraction() -> Reward:
    """Return a reward for abs(K + R - D - E) / length; empty sequences score 0."""
    def score(idr: str) -> float:
        if not idr:
            return 0.0
        pos = sum(idr.count(a) for a in "KR")
        neg = sum(idr.count(a) for a in "DE")
        return abs(pos - neg) / len(idr)

    return batchify(score)


def fraction_charged() -> Reward:
    """Return a reward for the fraction of D/E/K/R residues; empty sequences score 0."""
    return batchify(lambda idr: sum(idr.count(a) for a in "DEKR") / len(idr) if idr else 0.0)


def motif_count(pattern: str = r"[VILMF]K.E") -> Reward:
    """Return a reward counting non-overlapping regex matches per IDR.

    Args:
        pattern: Regex; defaults to the SUMOylation consensus [VILMF]K.E.

    Raises:
        re.error: If pattern is invalid; checked when the factory is called.
    """
    motif = re.compile(pattern)
    return batchify(lambda idr: float(len(motif.findall(idr))))

# Reward shaping example

def absolute_error(*, target: float) -> Shaping:
    """Return -abs(value - target): zero at the target, negative elsewhere."""
    return lambda value: -abs(value - target)
