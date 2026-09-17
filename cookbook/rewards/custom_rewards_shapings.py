"""Templates for in-process GRPO rewards and shaping; copy and adapt.

Set reward.name or shaping.name to /path/to/file.py:factory or package.module:factory
in the reward YAML. Put factory arguments, such as target, alongside name. See
cookbook/scripts/training/grpo/custom_reward.yaml for a complete configuration.

A reward factory returns a function mapping a list of sequences to one finite score
per sequence, in order, including empty strings. Use batchify to adapt a single-sequence
function. A shaping factory returns a function mapping one raw score to one finite
shaped score; IDiom applies the term's weight afterward.

These functions run in the training environment, so install dependencies there. Validate
arguments and load expensive resources in the factory, then reuse them when scoring.
"""

import re

from idiom.train.grpo.reward import Reward, Shaping, batchify


def net_charge_fraction() -> Reward:
    """Return a reward for abs(K + R - D - E) / length; empty sequences score 0."""

    # EDIT: replace this single-sequence calculation with your own measurement
    def score(idr: str) -> float:
        if not idr:  # In-process rewards must handle empty sequences themselves
            return 0.0
        pos = sum(idr.count(a) for a in "KR")
        neg = sum(idr.count(a) for a in "DE")
        return abs(pos - neg) / len(idr)

    return batchify(score)  # Wrap the scalar scorer to accept a list and preserve sequence order


def fraction_charged() -> Reward:
    """Return a reward for the fraction of D/E/K/R residues; empty sequences score 0."""
    # Return raw measurements; configure the target and shaping separately in YAML
    return batchify(lambda idr: sum(idr.count(a) for a in "DEKR") / len(idr) if idr else 0.0)


def motif_count(pattern: str = r"[VILMF]K.E") -> Reward:
    """Return a reward counting non-overlapping regex matches per IDR.

    Args:
        pattern: Regex; defaults to the SUMOylation consensus [VILMF]K.E.

    Raises:
        re.error: If pattern is invalid; checked when the factory is called.
    """
    # pattern is a factory argument supplied alongside reward.name in YAML
    motif = re.compile(pattern)  # Compile and validate once, before scoring starts
    return batchify(lambda idr: float(len(motif.findall(idr))))


# Reward shaping example


def absolute_error(*, target: float) -> Shaping:
    """Return -abs(value - target): zero at the target, negative elsewhere."""
    # target comes from shaping.target; the returned function receives each raw reward
    return lambda value: -abs(value - target)  # Highest score is 0; either side of the target is penalized
