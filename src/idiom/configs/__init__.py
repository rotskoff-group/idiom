"""Hydra config package.

The YAMLs beside this file are the shipped configs, composed by the idiom_* entrypoints and
overridable on the command line. This module makes them findable as package data, via
hydra.initialize_config_module("idiom.configs").

Everything a shipped config names is library code: the built-in rewards and the SAE feature reward
are modules under idiom.train.grpo.reward, named as dotted paths. A reward model that runs in its
own environment lives in cookbook/rewards/scorers/, and a term names it with an explicit `cmd`.
"""
