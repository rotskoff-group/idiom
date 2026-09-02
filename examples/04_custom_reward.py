"""Define a custom GRPO reward and show how to run RL post-training with it.

A reward is just f(idr: str) -> float registered by name. This script only defines and demonstrates
rewards (no training, no GPU, no weights); the printed command runs the actual RL.

    python examples/04_custom_reward.py
"""

from idiom.train.grpo.rewards import get_reward, register_group_reward, register_reward


@register_reward("aromatic_fraction")
def aromatic_fraction(idr: str) -> float:
    """Fraction of aromatic residues -- 'stickers' that drive condensate formation."""
    return sum(idr.count(a) for a in "FWY") / len(idr) if idr else 0.0


@register_group_reward("length_diversity")
def length_diversity(idrs: list[str], group_size: int) -> list[float]:
    """A group reward: score each completion by how far its length is from its group's mean.

    Group rewards see the whole GRPO group, so they can reward population-level properties
    (diversity, coverage) that no single sequence can satisfy alone.
    """
    mean = sum(len(s) for s in idrs) / max(len(idrs), 1)
    return [abs(len(s) - mean) / max(mean, 1.0) for s in idrs]


def main() -> None:
    examples = ["FFWYFFWY", "GSGSGSGS", "MEDSKVDNRPQ"]
    reward = get_reward("aromatic_fraction")
    for idr in examples:
        print(f"  aromatic_fraction({idr!r}) = {reward(idr):.3f}")

    group = ["AAAA", "AAAAAAAAAAAA", "AAAAAAAA"]
    print("  length_diversity(group) =", [round(x, 3) for x in length_diversity(group, len(group))])

    print("\nRun RL post-training with a custom reward (an external term names it):\n")
    print("  idiom_grpo init_from=/path/to/base.ckpt \\")
    print("    '+reward.external=[{enabled: true, weight: 1.0, name: aromatic_fraction, "
          "module: examples/04_custom_reward.py}]'\n")
    print("Reward toward an SAE feature signature (RL-SAE) instead:\n")
    print("  idiom_grpo init_from=/path/to/base.ckpt \\")
    print("    reward.rl_sae.enabled=true reward.rl_sae.signature=nucleolus")


if __name__ == "__main__":
    main()
