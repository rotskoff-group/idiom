"""Template for your own GRPO rewards and shaping. Copy this file into your project and edit it.

A reward reports a raw value; shaping says what a good value is. Both are written the same way --
a factory returning the thing it builds -- and both are named in a term by their "module:function"
path, so nothing here needs registering ahead of time:

    reward.terms='[{reward: "cookbook/rewards/custom_rewards.py:fraction_charged", weight: 1.0,
                    shaping: {name: gaussian, target: 0.25, width: 0.5}}]'

A factory takes the term's arguments and returns what runs every step: for a reward, a function
mapping the step's IDRs to one raw value each; for shaping, f(raw) -> float. Use lift when you have
a function that scores a single IDR, which is most of the time. Factories are called once, while
the config is validated, so validate arguments there and a bad setting fails in seconds.

A scorer whose dependencies cannot coexist with IDiom's runs in its own environment instead: see
scorers/ beside this file.

A reward whose optimum sits off the IDR distribution will be reached, and entropy and length terms
will not stop it -- rewarding hydrophobic composition, for instance, yields folded-looking
sequences that are no longer disordered.
"""

import re

from idiom.train.grpo.reward import Reward, Shaping, lift, tolerance


def net_charge_fraction() -> Reward:
    """Build a reward scoring the absolute net charge per residue (K/R positive, D/E negative).

    Natural IDR sets sit near 0.10; a target above that designs a polyelectrolyte.

    Returns:
        Reward: |net charge| divided by length, or 0.0 for an empty string.
    """
    def score(idr: str) -> float:
        if not idr:
            return 0.0
        pos = sum(idr.count(a) for a in "KR")
        neg = sum(idr.count(a) for a in "DE")
        return abs(pos - neg) / len(idr)

    return lift(score)


def fraction_charged() -> Reward:
    """Build a reward scoring the fraction of charged residues -- the FCR of the Das-Pappu diagram.

    FCR is how much charge there is and net_charge_fraction is how unbalanced it is; together they
    separate a polyampholyte from a polyelectrolyte. Natural IDR sets sit near 0.25.

    Returns:
        Reward: (D + E + K + R) divided by length, or 0.0 for an empty string.
    """
    return lift(lambda idr: sum(idr.count(a) for a in "DEKR") / len(idr) if idr else 0.0)


def motif_count(pattern: str = r"[VILMF]K.E") -> Reward:
    """Build a reward counting non-overlapping matches of a regex in an IDR.

    The default is psi-KxE, the SUMOylation consensus (hydrophobic-Lys-any-Glu); natural
    repression-domain IDRs carry about 0.6 per sequence. The NDSM variant extends it with a
    downstream acidic stretch, "[VILMF]K.E[DE]+", and is about 0.1 per sequence.

    This one takes an argument, which is what a term's reward mapping is for:

        reward: {name: "cookbook/rewards/custom_rewards.py:motif_count", pattern: "[VILMF]K.E[DE]+"}

    Args:
        pattern (str): The regex to count.

    Returns:
        Reward: Number of non-overlapping matches per IDR.

    Raises:
        ValueError: If the pattern does not compile. Raising in the factory reports it at config
            time rather than on the first training step.
    """
    motif = re.compile(pattern)  # compile now, so a bad pattern fails here
    return lift(lambda idr: float(len(motif.findall(idr))))


def one_sided(*, target: float, width: float = 1.0, direction: str = "above") -> Shaping:
    """Build a quadratic penalty on the wrong side of a threshold and no pressure on the right one.

    The three shipped rules -- quadratic, gaussian, identity -- all say *be here*, which is wrong
    whenever the objective is a threshold: an IDR that must stay expanded wants Rg >= 30 A, not
    Rg = 30 A. The acceptable side is flat, so it gives no gradient: pair this with a term that has
    a preference, or the policy settles just past the threshold.

    Args:
        target (float): The threshold; the penalty is 0 here and on the acceptable side.
        width (float): Tolerance as a fraction of the target, absolute when the target is 0.
        direction (str): "above" to accept values >= target, "below" to accept values <= target.

    Returns:
        Shaping: 0 on the acceptable side, -1 one tolerance into the wrong side, decreasing
            without bound beyond that.

    Raises:
        ValueError: If direction is neither "above" nor "below", or width is not positive.
    """
    if direction not in ("above", "below"):
        raise ValueError(f"one_sided direction must be 'above' or 'below', got {direction!r}")
    scale = tolerance(target, width)  # keeps width a fraction of a nonzero target, absolute at 0
    sign = 1.0 if direction == "above" else -1.0

    def shaping(value: float) -> float:
        deficit = sign * (target - value)  # positive only on the wrong side
        return -((deficit / scale) ** 2) if deficit > 0 else 0.0

    return shaping
