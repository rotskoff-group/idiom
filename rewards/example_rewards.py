"""Simple in-process GRPO rewards — copy and edit this file for your own.

A reward is f(idr: str) -> float: it receives the decoded IDR residue string and returns a scalar.
Register it with the register_reward("name") decorator; the entropy and length terms are added on
top by the config, so you usually only write the base signal.

Use it by adding an external term that names the reward and points at the file that registers it:

    reward.external:
      - {enabled: true, weight: 1.0, name: aromatic_fraction, module: rewards/example_rewards.py}

module accepts a *.py path (like here) or a dotted module path; it is imported before the reward is
looked up, so the decorators below run and register the rewards.
"""

from idiom.train.grpo.reward import register_reward


@register_reward("aromatic_fraction")
def aromatic_fraction(idr: str) -> float:
    """Return the fraction of aromatic residues (F/W/Y), e.g. to push toward sticker-rich IDRs.

    Args:
        idr (str): The decoded IDR residue string.

    Returns:
        float: Fraction of F/W/Y residues, or 0.0 for an empty string.
    """
    return sum(idr.count(a) for a in "FWY") / len(idr) if idr else 0.0


@register_reward("net_charge_fraction")
def net_charge_fraction(idr: str) -> float:
    """Return the absolute net charge per residue (K/R positive, D/E negative).

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
