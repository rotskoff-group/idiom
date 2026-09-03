"""Simple in-process GRPO rewards — copy and edit this file for your own.

A reward is f(idr: str) -> float: it receives the decoded IDR residue string and returns one raw
value in whatever units suit it. Register it with the register_reward("name") decorator and write
only that value — what counts as a good one is the term's shaping, set in the config, so the same
reward serves as a target, a guardrail, or a logged-only diagnostic.

Use it by adding a term that names the reward and points at the file that registers it:

    reward.terms:
      - {reward: net_charge_fraction, module: rewards/custom_rewards.py, weight: 1.0,
         shaping: {type: gaussian, target: 0.25, width: 0.5}}

module accepts a *.py path (like here) or a dotted module path; it is imported before the reward is
looked up, so the decorators below run and register the rewards. Omit shaping to use the raw value
directly, which for a fraction means pushing it toward 1.

Choose what you optimize with care: a reward whose optimum sits off the IDR distribution will be
reached, and the entropy and length guardrails will not stop it. Rewarding hydrophobic composition,
for instance, drives the policy to hydrophobic, folded-looking sequences that satisfy both
guardrails while no longer being disordered. The charge rewards below are safer: their targets sit
inside the natural IDR range, and charge is itself disorder-promoting.
"""

import re

from idiom.train.grpo.reward import register_reward

# Motif regexes as used in the IDiom manuscript: the SUMOylation consensus and its NDSM variant.
SUMO_PSI_KXE = re.compile(r"[VILMF]K.E")
NDSM = re.compile(r"[VILMF]K.E[DE]+")


@register_reward("net_charge_fraction")
def net_charge_fraction(idr: str) -> float:
    """Return the absolute net charge per residue (K/R positive, D/E negative).

    Natural IDR sets sit near 0.10; a gaussian target above that designs a polyelectrolyte. Charge
    is disorder-promoting, so pushing this up keeps the sequence on the IDR distribution.

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
    """Return the fraction of charged residues (D/E/K/R) — the FCR of the Das-Pappu diagram.

    FCR and net charge are the two standard axes for classifying an IDR's conformational state:
    FCR is how much charge there is, net_charge_fraction is how unbalanced it is, so a strong
    polyampholyte (high FCR, near-zero net charge) and a polyelectrolyte (high FCR, high net
    charge) are told apart only by the pair. Natural IDR sets sit near 0.25; raising it expands
    the chain, and charge is disorder-promoting, so the optimum stays on the IDR distribution.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: (D + E + K + R) divided by length, or 0.0 for an empty string.
    """
    return sum(idr.count(a) for a in "DEKR") / len(idr) if idr else 0.0


@register_reward("sumo_motif_count")
def sumo_motif_count(idr: str) -> float:
    """Return the number of psi-KxE SUMOylation consensus motifs in an IDR.

    psi-KxE (hydrophobic-Lys-any-Glu, regex [VILMF]K.E) is the SUMO consensus; SUMOylation and the
    corepressor recruitment that follows is a common route to transcriptional repression. Because
    SUMO-mediated interactions depend on avidity, motif *density* matters, which is why this counts
    rather than reporting presence. Natural repression-domain IDRs carry about 0.6 per sequence.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Number of non-overlapping matches.
    """
    return float(len(SUMO_PSI_KXE.findall(idr)))


@register_reward("ndsm_motif_count")
def ndsm_motif_count(idr: str) -> float:
    """Return the number of NDSM (negatively charged amino acid-dependent SUMOylation motif) hits.

    The NDSM variant extends psi-KxE with a downstream acidic stretch ([VILMF]K.E[DE]+), which
    raises SUMOylation efficiency. Natural repression-domain IDRs carry about 0.1 per sequence.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Number of non-overlapping matches.
    """
    return float(len(NDSM.findall(idr)))
