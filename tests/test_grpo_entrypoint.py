"""GRPO entrypoint tests: composite reward + build() wiring."""

import math
import re
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from idiom.model import IDiomTransformer, ModelConfig
from idiom.train.grpo import LitGRPO
from idiom.train.grpo.data import PromptDataset
from idiom.train.grpo.reward import REWARD_ALIASES, build_terms
from idiom.train.grpo.train_grpo import build, build_reward

TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=64)


def _ckpt(tmp_path):
    """Save a pretrained checkpoint with its ModelConfig."""
    m = IDiomTransformer(TINY)
    ckpt = tmp_path / "pretrained.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in m.state_dict().items()},
            "hyper_parameters": {"model_cfg": asdict(TINY)},
        },
        ckpt,
    )
    return ckpt


def _reward_cfg(**over):
    """The configs/grpo.yaml reward structure, naming a fixture reward by its path."""
    base = {
        "terms": [
            {
                "reward": "length",
                "weight": 2.0,
                "shaping": {"name": "quadratic", "target": 100, "width": 0.1},
            },
            {"reward": "tests.reward_fixtures:fraction_proline", "weight": 1.0},
        ],
    }
    base.update(over)
    return OmegaConf.create(base)


def test_build_reward_composes():
    terms = build_reward(_reward_cfg())
    idr = "P" * 100  # 100% proline, and length exactly on the target
    totals, _ = terms([idr], 1)
    assert abs(totals[0] - 1.0) < 1e-6


def test_the_shipped_aliases_are_the_whole_menu():
    from idiom.train.grpo.reward import REWARD_ALIASES, SHAPING_ALIASES, entropy, length

    assert set(REWARD_ALIASES) == {"entropy", "length", "external_scorer", "sae_signature"}
    assert set(SHAPING_ALIASES) == {"quadratic", "gaussian", "identity"}
    assert entropy()(["AAAA"]) == [0.0] and length()(["AAAA"]) == [4.0]


def test_every_alias_resolves_to_a_factory():
    from idiom.train.grpo.reward import REWARD_ALIASES, SHAPING_ALIASES, load_callable

    for name, path in {**REWARD_ALIASES, **SHAPING_ALIASES}.items():
        assert callable(load_callable(path)), name


def test_build_wires_module_and_prompts(tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "init_from": str(_ckpt(tmp_path)),
            "grpo": {
                "group_size": 2,
                "max_new_tokens": 6,
                "lr": 5e-6,
                "beta_kl": 0.02,
                "eps_clip": 0.2,
                "temperature": 1.0,
                "top_k": None,
                "top_p": None,
                "normalize_advantage": True,
            },
            "reward": _reward_cfg(),
            "prompts": {"mode": "unprompted", "n": 16, "fasta": None, "n_per": 1, "batch_size": 4},
        }
    )
    lit, ds = build(cfg)
    assert isinstance(lit, LitGRPO) and lit.group_size == 2 and lit.cfg == TINY
    assert isinstance(ds, PromptDataset) and len(ds) == 16


def test_build_rejects_unknown_prompt_mode(tmp_path):
    import pytest

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "init_from": str(_ckpt(tmp_path)),
            "grpo": {"group_size": 2, "max_new_tokens": 6},
            "reward": _reward_cfg(),
            "prompts": {"mode": "idp", "n": 16, "fasta": None, "n_per": 1, "batch_size": 4},
        }
    )
    with pytest.raises(ValueError, match="'prompted' or 'unprompted'"):
        build(cfg)


def test_nothing_shipped_names_a_path_outside_the_package():
    import idiom.configs

    cfgdir = Path(idiom.configs.__file__).parent
    for yaml in sorted(cfgdir.glob("*.yaml")):
        # Avoid resolving Hydra runtime interpolations such as ${now:...}
        blob = OmegaConf.to_container(OmegaConf.load(yaml), resolve=False)
        for value in _strings(blob):
            if value.startswith("${"):
                continue  # an env/oc interpolation, not a path
            assert "cookbook" not in value, f"{yaml.name} names repository material: {value!r}"
            assert not re.search(r"(^|\s)[./]*/?[\w./-]+\.py(\s|$)", value), (
                f"{yaml.name} names a script path: {value!r}"
            )


def _strings(obj):
    """Yield every string anywhere in a nested config."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def test_the_sae_reward_ships_with_its_signatures():
    from idiom.train.grpo.reward import sae_feature

    assert (Path(sae_feature.__file__).parent / "sae_signatures.json").is_file()
    assert sae_feature._featuresets(sae_feature.DEFAULT_FEATURES, sae_feature.DEFAULT_CASE), (
        "no signatures in the shipped file"
    )
    assert "sae_signature" in REWARD_ALIASES


def test_the_shipped_objective_is_empty():
    import idiom.configs

    cfg = OmegaConf.load(Path(idiom.configs.__file__).parent / "grpo.yaml")
    assert list(cfg.reward.terms) == []
    assert set(cfg.reward) == {"terms"}, "the reward config is a terms list, nothing else"
    with pytest.raises(ValueError, match="reward.terms is empty"):
        build_terms(cfg.reward)


def test_the_rewards_the_library_registers_need_no_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    terms = [
        {"reward": "entropy", "weight": 1.0, "shaping": {"name": "quadratic", "target": 3.65, "width": 0.2}},
        {"reward": "length", "weight": 1.0, "shaping": {"name": "quadratic", "target": 100, "width": 1.0}},
    ]
    totals, _ = build_reward(OmegaConf.create({"terms": terms}))(["P" * 100], 1)
    assert math.isfinite(totals[0])
