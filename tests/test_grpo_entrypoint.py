"""P4 GRPO entrypoint tests (CPU-only): composite reward + build() wiring."""

import math
import re
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from idiom.model import IDiomTransformer, ModelConfig
from idiom.train.grpo import LitGRPO
from idiom.train.grpo.data import PromptDataset
from idiom.train.grpo.reward import parse_terms
from idiom.train.grpo.train_grpo import build, build_reward

TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=64)


def _ckpt(tmp_path):
    """Self-describing pretrained ckpt for GRPO to warm-start from."""
    m = IDiomTransformer(TINY)
    ckpt = tmp_path / "pretrained.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in m.state_dict().items()},
         "hyper_parameters": {"model_cfg": asdict(TINY)}},
        ckpt,
    )
    return ckpt


def _reward_cfg(**over):
    """The configs/grpo.yaml reward structure, using the registered fraction_proline."""
    base = {
        "module": None,
        "terms": [
            {"reward": "length", "weight": 2.0,
             "shaping": {"type": "quadratic", "target": 100, "width": 0.1}},
            {"reward": "fraction_proline", "weight": 1.0},
        ],
    }
    base.update(over)
    return OmegaConf.create(base)


def test_build_reward_composes():
    terms = build_reward(_reward_cfg())
    idr = "P" * 100  # 100% proline, and length exactly on the target
    totals, _ = terms([idr], 1)
    # fraction_proline = 1.0 at weight 1.0; the quadratic length penalty is 0 at the target
    assert abs(totals[0] - 1.0) < 1e-6


def test_the_library_defines_only_the_guardrails():
    """entropy and length register on import; nothing else is built in.

    Two things ride on this. `pip install git+...` is usable because the shipped config names the
    guardrails directly and needs nothing on disk beside it. And the library takes no view on what
    you should design for: every other reward is repository material a run points a term at, so
    adding one here is a deliberate statement that it belongs to everybody.
    """
    from idiom.train.grpo.reward import REWARD_REGISTRY, Batch, get_reward

    assert get_reward("entropy")(["AAAA"], Batch()) == [0.0]
    assert get_reward("length")(["AAAA"], Batch()) == [4.0]
    # the fixtures module registers two more; nothing else may creep in
    assert set(REWARD_REGISTRY) - {"fraction_proline", "fraction_alanine"} == {"entropy", "length"}


def test_build_wires_module_and_prompts(tmp_path):
    cfg = OmegaConf.create({
        "seed": 0,
        "init_from": str(_ckpt(tmp_path)),  # GRPO warm-starts; arch read from this ckpt
        "grpo": {"group_size": 2, "max_new_tokens": 6, "lr": 5e-6, "beta_kl": 0.02,
                 "eps_clip": 0.2, "temperature": 1.0, "top_k": None, "top_p": None,
                 "normalize_advantage": True},
        "reward": _reward_cfg(),
        "prompts": {"mode": "unprompted", "n": 16, "fasta": None, "n_per": 1, "batch_size": 4},
    })
    lit, ds = build(cfg)
    assert isinstance(lit, LitGRPO) and lit.group_size == 2 and lit.cfg == TINY
    assert isinstance(ds, PromptDataset) and len(ds) == 16


def test_build_rejects_unknown_prompt_mode(tmp_path):
    import pytest

    cfg = OmegaConf.create({
        "seed": 0,
        "init_from": str(_ckpt(tmp_path)),
        "grpo": {"group_size": 2, "max_new_tokens": 6},
        "reward": _reward_cfg(),
        "prompts": {"mode": "idp", "n": 16, "fasta": None, "n_per": 1, "batch_size": 4},
    })
    with pytest.raises(ValueError, match="'prompted' or 'unprompted'"):
        build(cfg)


def test_nothing_shipped_names_a_path_outside_the_package():
    """The shipped configs must name only library code -- no filesystem paths at all.

    This is what makes every install equivalent. A reward model that runs in its own environment is
    a standalone program in the repository (cookbook/rewards/scorers/), named by an explicit cmd in
    the run that wants it; if one ever leaked into the shipped menu, a pip install would carry a
    config pointing at a file it does not have.
    """
    import idiom.configs

    cfgdir = Path(idiom.configs.__file__).parent
    for yaml in sorted(cfgdir.glob("*.yaml")):
        # resolve=False: the hydra block carries ${now:...}, and nothing in the reward config
        # interpolates any more -- which is the point of the check.
        blob = OmegaConf.to_container(OmegaConf.load(yaml), resolve=False)
        for value in _strings(blob):
            if value.startswith("${"):
                continue  # an env/oc interpolation, not a path
            assert "cookbook" not in value, f"{yaml.name} names repository material: {value!r}"
            assert not re.search(r"(^|\s)[./]*/?[\w./-]+\.py(\s|$)", value), \
                f"{yaml.name} names a script path: {value!r}"


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
    """The SAE feature reward is a module, and the released SAE's signatures sit beside it."""
    from idiom.train.grpo.reward import sae_feature

    assert (Path(sae_feature.__file__).parent / "sae_signatures.json").is_file()
    assert sae_feature._signature_names(), "no signatures registered from the shipped file"


def test_the_shipped_objective_is_the_guardrails_and_nothing_else():
    """Out of the box a run optimizes only entropy and length, and `add` is the extension point.

    The config ships inert on purpose: what a run designs for is named at launch (see
    cookbook/scripts/), so nothing is silently optimizing on somebody's behalf.
    """
    import idiom.configs

    cfg = OmegaConf.load(Path(idiom.configs.__file__).parent / "grpo.yaml")
    assert [t["reward"] for t in cfg.reward.terms] == ["entropy", "length"]
    assert list(cfg.reward.add) == [] and dict(cfg.reward.presets) == {}
    assert [s.label for s in parse_terms(cfg.reward)] == ["entropy", "length"]


def test_shipped_config_enabled_terms_need_no_repository(tmp_path, monkeypatch):
    """The terms that are ON by default must build from the package alone.

    A disabled term may name anything, but an enabled one runs on step 1, so if the guardrails
    needed a file outside the wheel, idiom_train_grpo would be dead on arrival for a pip install. Running
    from an unrelated working directory is the check: nothing may resolve relative to the cwd.
    """
    import idiom.configs

    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.load(Path(idiom.configs.__file__).parent / "grpo.yaml")
    enabled = [t for t in OmegaConf.to_container(cfg.reward, resolve=True)["terms"]
               if t.get("enabled", True)]
    assert [t["reward"] for t in enabled] == ["entropy", "length"]
    assert all(t.get("module") is None for t in enabled), "an enabled term must need no module"
    totals, _ = build_reward(OmegaConf.create({"module": None, "terms": enabled}))(["P" * 100], 1)
    assert math.isfinite(totals[0])
