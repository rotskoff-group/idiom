"""Example custom GRPO rewards — copy/edit this file for your own.

A reward is ``f(idr: str) -> float``: it receives the decoded IDR residue string and returns a
scalar. Register it with ``@register_reward("name")``; the optional shaping / length / entropy
terms are added on top by the config, so you usually only write the base signal.

Use it by pointing GRPO at this file and selecting the name::

    idiom_grpo init_from=... reward.module=rewards/example_rewards.py reward.name=aromatic_fraction

``reward.module`` accepts a ``*.py`` path (like here) or a dotted module path; it's imported
before the reward is looked up, so the decorators below run and register the rewards.
"""

from idiom.train.grpo.rewards import register_reward


@register_reward("aromatic_fraction")
def aromatic_fraction(idr: str) -> float:
    """Fraction of aromatic residues (F/W/Y) — e.g. to push toward sticker-rich IDRs."""
    return sum(idr.count(a) for a in "FWY") / len(idr) if idr else 0.0


@register_reward("net_charge_fraction")
def net_charge_fraction(idr: str) -> float:
    """|net charge| per residue (K/R positive, D/E negative)."""
    if not idr:
        return 0.0
    pos = sum(idr.count(a) for a in "KR")
    neg = sum(idr.count(a) for a in "DE")
    return abs(pos - neg) / len(idr)
