"""User-editable GRPO reward content, shipped inside the package so it survives a pip install.

Three kinds of thing live here, and only the first is ever imported by IDiom:

    custom_rewards.py    in-process rewards: f(idr) -> float, registered with @register_reward and
                         named by a term's `module`. The entropy and length guardrails the shipped
                         config enables live here too, so every in-process reward is in one file.
    external_rewards/    standalone scorer programs, each carrying its own environment in a PEP 723
                         header and run as a subprocess (`uv run --script <path>`). They are never
                         imported -- their dependencies do not coexist with IDiom's.
    rl_sae_reward/       the SAE feature-code reward and the signature file it reads; imported on
                         demand, which keeps torch and the SAE lens out of runs that leave it off.

configs/grpo.yaml addresses these through the ${idiom_rewards:...} resolver, so the paths work from
any working directory and in any install. To edit them, copy the tree into your own project:

    idiom_rewards --init          # writes ./rewards/, then point `module` / `cmd` at your copy
"""

from pathlib import Path

REWARDS_DIR = Path(__file__).resolve().parent


def rewards_path(relative: str = "") -> Path:
    """Return the absolute path of a shipped reward file.

    Args:
        relative (str): Path relative to this package, e.g. "external_rewards/sparrow.py".

    Returns:
        Path: The absolute path, whether IDiom is installed as a wheel or from a clone.
    """
    return REWARDS_DIR / relative if relative else REWARDS_DIR


__all__ = ["REWARDS_DIR", "rewards_path"]
