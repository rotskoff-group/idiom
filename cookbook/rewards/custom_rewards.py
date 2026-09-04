"""Template for your own GRPO rewards. Copy this file into your project and edit it.

A reward is a function of the decoded IDR residue string returning one raw value in whatever units
suit it. What counts as a good value is the term's shaping, set in the config. The library defines
only `entropy` and `length`; everything below is an example to edit or throw away.

There are two ways to point a term at a function in this process.

1. A file or module that registers names, named by the term's `module`:

       reward.add='[{reward: fraction_charged, module: /path/to/custom_rewards.py, weight: 1.0,
                     shaping: {type: gaussian, target: 0.25, width: 0.5}}]'

   `module` takes a *.py path (as here) or a dotted module name, and is imported before the reward
   name is looked up, so the @register_reward decorators below run.

2. Any importable callable, named directly as "module:function", with no decorator:

       reward.add='[{reward: "mypackage.scoring:score_idr", weight: 1.0}]'

Either way nothing is imported until the term is used, and a name that cannot be resolved fails
while the config is parsed.

A scorer whose dependencies cannot coexist with IDiom's goes in its own environment instead: see
scorers/ beside this file. Shaping registers the same way, in custom_shaping.py.

Note that a reward whose optimum sits off the IDR distribution will be reached, and the entropy and
length guardrails will not stop it -- rewarding hydrophobic composition, for instance, yields
folded-looking sequences that are no longer disordered.
"""

import re

from idiom.train.grpo.reward import Batch, register_reward


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


# --- the batched form ---------------------------------------------------------------------------
# Register with batched=True (or set `batched: true` on a "module:function" term) to receive the
# whole step's completions at once. Use it when scoring costs less per batch than per sequence -- a
# GPU forward pass, a vectorized model -- or, as here, when a sequence can only be scored against
# the others it was sampled with.


@register_reward("charge_rank_in_group", batched=True)
def charge_rank_in_group(idrs: list[str], batch: Batch) -> list[float]:
    """Return each completion's within-group rank on FCR, scaled to [0, 1].

    GRPO samples group_size completions per prompt, so consecutive runs of that many entries belong
    to one group. This demonstrates the batched signature rather than being a reward to reach for.

    Args:
        idrs (list[str]): The decoded IDR residue strings for this step.
        batch (Batch): What the reward knows about the batch; carries group_size.

    Returns:
        list[float]: One value per idr, in the order they were given.
    """
    out = [0.0] * len(idrs)
    g = max(1, batch.group_size)
    for start in range(0, len(idrs), g):
        group = list(range(start, min(start + g, len(idrs))))
        order = sorted(group, key=lambda i: fraction_charged(idrs[i]))
        for rank, i in enumerate(order):
            out[i] = rank / max(1, len(group) - 1)
    return out
