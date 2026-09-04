"""Hydra config package.

The YAMLs beside this file are the shipped configs, composed by the idiom_* entrypoints and
overridable on the command line. This module exists so they can be found as package data
(hydra.initialize_config_module("idiom.configs")) in any install.

Everything a shipped config can name is library code: the built-in rewards and the SAE feature
reward are modules under idiom.train.grpo.reward, named as dotted paths. Nothing here resolves a
filesystem path, because nothing shipped needs one -- a reward model that runs in its own
environment is a standalone program in the repository's cookbook/rewards/scorers/, and a term names
it with an explicit `cmd`.
"""
