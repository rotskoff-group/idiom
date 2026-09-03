"""P4 GRPO entrypoint tests (CPU-only): composite reward + build() wiring."""

from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from idiom.model import IDiomTransformer, ModelConfig
from idiom.train.grpo import LitGRPO
from idiom.train.grpo.data import PromptDataset
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


def test_shipped_example_rewards_register():
    # the shipped example file registers via the reward.module mechanism, from wherever it is
    # installed (it lives inside the package now, so this holds for a wheel too)
    from idiom.configs import rewards_path
    from idiom.train.grpo.reward import Batch, get_reward, import_module_spec

    import_module_spec(str(rewards_path("custom_rewards.py")))
    # 3 of 4 are aromatic
    assert get_reward("net_charge_fraction")(["KKDE"], Batch()) == [0.0]
    assert get_reward("fraction_charged")(["KKDE"], Batch()) == [1.0]


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


def test_shipped_config_reward_paths_resolve():
    """Every ${idiom_rewards:...} path in the shipped config must exist in this install.

    The reward content moved inside the package so a pip install keeps it; a stale path here would
    only surface when a user switched a term on, which is exactly the wrong time to find out.
    """
    from omegaconf import OmegaConf

    import idiom.configs  # noqa: F401 - registers the resolver
    from idiom.configs import rewards_path

    cfg = OmegaConf.load(Path(idiom.configs.__file__).parent / "grpo.yaml")
    resolved = OmegaConf.to_container(cfg.reward, resolve=True)
    checked = []
    for term in resolved["terms"]:
        for value in (term.get("module"), term.get("cmd")):
            if not value or not value.startswith(("/", "uv run --script /")):
                continue
            path = Path(value.split("uv run --script ")[-1].split()[0])
            assert path.is_file(), f"{path} is referenced by the shipped config but not installed"
            checked.append(path)
    assert len(checked) >= 7  # custom_rewards + six scorers
    assert rewards_path("rl_sae_reward/idiomsae-300M-L18-k32.json").is_file()
