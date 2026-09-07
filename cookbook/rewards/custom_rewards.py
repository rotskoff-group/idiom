"""Templates for in-process GRPO rewards and shaping; copy and adapt.

Factories receive config arguments and return batch rewards or scalar shaping functions.
Use batchify for single-IDR scorers and "module:function" paths to name custom factories.
Validate settings in the factory. See cookbook/rewards/README.md for configuration.
"""

import re

from idiom.train.grpo.reward import Reward, Shaping, batchify, tolerance


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


def one_sided(*, target: float, width: float = 1.0, direction: str = "above") -> Shaping:
    """Return a quadratic penalty outside an acceptable threshold.

    Args:
        target: Threshold where the penalty reaches 0.
        width: Positive fractional tolerance; absolute when target is 0.
        direction: "above" accepts values >= target; "below" accepts values <= target.

    Returns:
        A function scoring 0 on the accepted side and -1 one tolerance outside it.

    Raises:
        ValueError: If direction is invalid or width is not positive.
    """
    if direction not in ("above", "below"):
        raise ValueError(f"one_sided direction must be 'above' or 'below', got {direction!r}")
    scale = tolerance(target, width)
    sign = 1.0 if direction == "above" else -1.0

    def shaping(value: float) -> float:
        deficit = sign * (target - value)  # positive only on the wrong side
        return -((deficit / scale) ** 2) if deficit > 0 else 0.0

    return shaping
