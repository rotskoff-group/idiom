"""Template for your own GRPO rewards and shaping. Copy this file into your project and edit it.

A reward reports a raw value; shaping says what a good value is. Both register here, so one term's
`module` brings in both and `reward.module` stays null.
"""

import re

from idiom.train.grpo.reward import register_reward, register_shaping, tolerance


@register_reward("net_charge_fraction")
def net_charge_fraction(idr: str) -> float:
    """Return the absolute net charge per residue (K/R positive, D/E negative).

    Natural IDR sets sit near 0.10; a target above that designs a polyelectrolyte.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: |net charge| divided by length, or 0.0 for an empty string.
    """
    if not idr:
        return 0.0
    pos = sum(idr.count(a) for a in "KR")
    neg = sum(idr.count(a) for a in "DE")
    return abs(pos - neg) / len(idr)


@register_reward("fraction_charged")
def fraction_charged(idr: str) -> float:
    """Return the fraction of charged residues (D/E/K/R) -- the FCR of the Das-Pappu diagram.

    FCR is how much charge there is and net_charge_fraction is how unbalanced it is; together they
    separate a polyampholyte from a polyelectrolyte. Natural IDR sets sit near 0.25.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: (D + E + K + R) divided by length, or 0.0 for an empty string.
    """
    return sum(idr.count(a) for a in "DEKR") / len(idr) if idr else 0.0


@register_reward("sumo_motif_count")
def sumo_motif_count(idr: str) -> float:
    """Return the number of psi-KxE SUMOylation consensus motifs in an IDR.

    psi-KxE is hydrophobic-Lys-any-Glu, the regex [VILMF]K.E. Natural repression-domain IDRs carry
    about 0.6 per sequence.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Number of non-overlapping matches.
    """
    psi_kxe = r"[VILMF]K.E"
    return float(len(re.findall(psi_kxe, idr)))


@register_reward("ndsm_motif_count")
def ndsm_motif_count(idr: str) -> float:
    """Return the number of NDSM (negatively charged amino acid-dependent SUMOylation motif) hits.

    The NDSM variant extends psi-KxE with a downstream acidic stretch, the regex [VILMF]K.E[DE]+.
    Natural repression-domain IDRs carry about 0.1 per sequence.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Number of non-overlapping matches.
    """
    ndsm = r"[VILMF]K.E[DE]+"
    return float(len(re.findall(ndsm, idr)))


# custom reward shaping


@register_shaping("one_sided")
def one_sided(*, target: float, width: float = 1.0, direction: str = "above"):
    """Build a quadratic penalty on the wrong side of a threshold and no pressure on the right one.

    The acceptable side is flat, so it gives no gradient: pair this with a term that has a
    preference, or the policy settles just past the threshold.

    Args:
        target (float): The threshold; the penalty is 0 here and on the acceptable side.
        width (float): Tolerance as a fraction of the target, absolute when the target is 0.
        direction (str): "above" to accept values >= target, "below" to accept values <= target.

    Returns:
        Callable[[float], float]: 0 on the acceptable side, -1 one tolerance into the wrong side,
            decreasing without bound beyond that.

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
