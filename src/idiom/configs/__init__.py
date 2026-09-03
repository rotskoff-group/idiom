"""Hydra config package, and the resolvers its configs use.

Importing this registers ${idiom_rewards:...}, which resolves a path inside the repository's
top-level rewards/ directory. It is anchored to the installed package rather than to the working
directory, so a term keeps working wherever a run is launched from.

That anchor only exists for an editable install (`uv sync`, or `pip install -e .` from a clone),
which is how IDiom is meant to be used: the reward content is repository material you edit, not
library code. A non-editable install has no rewards/ to point at, and the resolver says so.
"""

from pathlib import Path

from omegaconf import OmegaConf

import idiom


def repo_root() -> Path:
    """Return the repository root of an editable install.

    Returns:
        Path: The directory holding rewards/, cookbook/, and pyproject.toml.

    Raises:
        RuntimeError: If IDiom was not installed from a clone, so there is no repository to find.
    """
    root = Path(idiom.__file__).resolve().parents[2]
    if not (root / "rewards").is_dir():
        raise RuntimeError(
            f"the shipped rewards live in the repository's rewards/ directory, which is not next to "
            f"this install ({root}). IDiom is used from a clone: git clone the repository and run "
            f"`uv sync` (or `pip install -e .`), then launch from there."
        )
    return root


def rewards_path(relative: str = "") -> Path:
    """Return the absolute path of a file in the repository's rewards/ directory.

    Args:
        relative (str): Path relative to rewards/, e.g. "external_rewards/sparrow.py".

    Returns:
        Path: The absolute path.
    """
    rewards = repo_root() / "rewards"
    return rewards / relative if relative else rewards


if not OmegaConf.has_resolver("idiom_rewards"):
    OmegaConf.register_new_resolver("idiom_rewards", lambda rel: str(rewards_path(rel)))
