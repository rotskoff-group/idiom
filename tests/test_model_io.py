"""Unified model loading (CPU-only): self-describing checkpoints + format-agnostic load_model."""

from dataclasses import asdict

import pytest
import torch

from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.io import config_from_checkpoint, load_model, load_pretrained

CFG = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)


def _ckpt(tmp_path, cfg=CFG, *, with_cfg=True):
    """A Lightning-style ckpt; self-describing (hyper_parameters.model_cfg) unless with_cfg=False."""
    m = IDiomTransformer(cfg)
    obj = {"state_dict": {f"model.{k}": v for k, v in m.state_dict().items()}}
    if with_cfg:
        obj["hyper_parameters"] = {"model_cfg": asdict(cfg)}
    ckpt = tmp_path / "m.ckpt"
    torch.save(obj, ckpt)
    return m, ckpt


def test_config_from_checkpoint_reads_hparams(tmp_path):
    _, ckpt = _ckpt(tmp_path)
    assert config_from_checkpoint(ckpt) == CFG


def test_config_from_checkpoint_errors_without_stored_cfg(tmp_path):
    _, ckpt = _ckpt(tmp_path, with_cfg=False)
    with pytest.raises(ValueError, match="no stored ModelConfig"):
        config_from_checkpoint(ckpt)


def test_load_pretrained_reads_arch_from_checkpoint(tmp_path):
    m, ckpt = _ckpt(tmp_path)
    loaded = load_pretrained(ckpt)  # no cfg — arch is read from the ckpt
    assert loaded.cfg == CFG
    for a, b in zip(m.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    assert not loaded.training  # eval mode


def test_load_model_dispatches_ckpt_and_dir(tmp_path):
    from idiom import IDiom

    _, ckpt = _ckpt(tmp_path)
    model_ck, cfg_ck = load_model(ckpt)  # .ckpt path
    assert cfg_ck == CFG and isinstance(model_ck, IDiomTransformer)

    rel = tmp_path / "rel"
    IDiom.from_lightning_checkpoint(ckpt).save_pretrained(rel)  # released dir
    model_dir, cfg_dir = load_model(rel)
    assert cfg_dir == CFG
    for a, b in zip(model_ck.state_dict().values(), model_dir.state_dict().values()):
        assert torch.equal(a, b)


def test_lit_modules_persist_model_cfg():
    from idiom.train.grpo.lit_grpo import LitGRPO
    from idiom.train.lit_autoregressive import LitAutoregressive

    assert LitAutoregressive(CFG).hparams["model_cfg"] == asdict(CFG)
    assert LitGRPO(CFG, reward_fn=lambda s: 0.0).hparams["model_cfg"] == asdict(CFG)
