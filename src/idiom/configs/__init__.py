"""Hydra config package, and the resolvers its configs use.

Importing this registers ${idiom_rewards:...}, which resolves a path relative to the shipped
idiom.rewards package. Configs use it instead of a repo-relative path so a term keeps working from
any working directory, whether IDiom was installed as a wheel or from a clone.
"""

from omegaconf import OmegaConf

from idiom.rewards import rewards_path

if not OmegaConf.has_resolver("idiom_rewards"):
    OmegaConf.register_new_resolver("idiom_rewards", lambda rel: str(rewards_path(rel)))
