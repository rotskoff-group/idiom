"""Template for your own GRPO shaping. Copy this file into your project and edit it.

A reward reports a raw value; shaping says what a good value is. The library ships `quadratic`,
`gaussian`, and identity -- all of which say "be here" -- so a threshold, a band, or anything else
is a rule of your own. A factory takes the spec's parameters and returns f(raw) -> float; validate
in the factory, which runs once at config time, rather than in the rule it returns.

Name the rule in a term's `shaping.type` and this file in `reward.module`, which is imported before
every term:

    idiom_train_grpo reward.module=cookbook/rewards/custom_shaping.py \
      reward.add='[{reward: fraction_charged, module: cookbook/rewards/custom_rewards.py,
                    weight: 1.0, shaping: {type: one_sided, target: 0.30, width: 0.5}}]'

A term takes one `module`, already spent on the reward here. If your reward and your shaping share
one file, that file in the term's own `module` is enough.
"""

from idiom.train.grpo.reward import register_shaping, tolerance


@register_shaping("one_sided")
def one_sided(*, target: float, width: float = 1.0, direction: str = "above"):
    """Build a quadratic penalty on the wrong side of a threshold and no pressure on the right one.

    An IDR that must stay expanded wants Rg >= 30 A, not Rg == 30 A, and a two-sided rule spends
    optimization pressure pulling improvements back to the target. Flat also means no gradient, so
    pair this with a term that has a preference or the policy settles just past the threshold.

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
    scale = tolerance(target, width)
    sign = 1.0 if direction == "above" else -1.0

    def shaping(value: float) -> float:
        deficit = sign * (target - value)  # positive only on the wrong side
        return -((deficit / scale) ** 2) if deficit > 0 else 0.0

    return shaping
