"""P4 GRPO entrypoint tests (CPU-only): composite reward + build() wiring."""

from dataclasses import asdict

import torch
from omegaconf import OmegaConf

from idiom.model import IDiomTransformer, ModelConfig
from idiom.train.grpo import LitGRPO
from idiom.train.grpo.data import PromptDataset
from idiom.train.grpo.train_grpo import build, build_reward_terms

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
    """The configs/grpo.yaml reward shape, using the suite's registered fraction_proline."""
    base = {
        "module": None,
        "entropy": {"enabled": False, "weight": 1.0, "target_entropy": 2.7, "width": 0.2},
        "length": {"enabled": True, "weight": 2.0, "target_length": 100, "width": 0.1},
        "rl_sae": {"enabled": False, "weight": 1.0, "signature": "nucleolus"},
        "external": [{"enabled": True, "weight": 1.0, "name": "fraction_proline"}],
    }
    base.update(over)
    return OmegaConf.create(base)


def test_build_reward_composes():
    terms = build_reward_terms(_reward_cfg())
    idr = "P" * 100  # 100% proline, and length exactly on the target
    totals, _ = terms([idr], 1)
    # fraction_proline = 1.0 at weight 1.0; the quadratic length penalty is 0 at the target
    assert abs(totals[0] - 1.0) < 1e-6


def test_shipped_example_rewards_register():
    # the example file in rewards/ registers via the reward.module mechanism
    from pathlib import Path

    from idiom.train.grpo.reward import get_reward
    from idiom.train.grpo.reward.compose_reward import _register_custom_rewards

    path = Path(__file__).resolve().parents[1] / "rewards" / "example_rewards.py"
    _register_custom_rewards(str(path))
    assert get_reward("aromatic_fraction")("FWYA") == 0.75  # 3 of 4 are aromatic


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
