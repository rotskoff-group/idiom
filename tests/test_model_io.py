"""Unified model loading: self-describing checkpoints + format-agnostic load_model."""

from dataclasses import asdict

import pytest
import torch

from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.io import config_from_checkpoint, load_model, load_pretrained

CFG = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)


def _ckpt(tmp_path, cfg=CFG, *, with_cfg=True):
    """Save a Lightning checkpoint, optionally omitting its ModelConfig."""
    m = IDiomTransformer(cfg)
    obj = {"state_dict": {f"model.{k}": v for k, v in m.state_dict().items()}}
    if with_cfg:
        obj["hyper_parameters"] = {"model_cfg": asdict(cfg)}
    ckpt = tmp_path / "m.ckpt"
    torch.save(obj, ckpt)
    return m, ckpt


def test_config_from_checkpoint_reads_hparams(tmp_path):
    """Verify recovery of model configuration from checkpoint hyperparameters."""
    _, ckpt = _ckpt(tmp_path)
    assert config_from_checkpoint(ckpt) == CFG


def test_config_from_checkpoint_errors_without_stored_cfg(tmp_path):
    """Verify rejection of checkpoints without a stored model configuration."""
    _, ckpt = _ckpt(tmp_path, with_cfg=False)
    with pytest.raises(ValueError, match="no stored ModelConfig"):
        config_from_checkpoint(ckpt)


def test_load_pretrained_reads_arch_from_checkpoint(tmp_path):
    """Verify checkpoint architecture, weights, and evaluation mode after loading."""
    m, ckpt = _ckpt(tmp_path)
    loaded, cfg = load_pretrained(ckpt)
    assert loaded.cfg == CFG
    assert cfg == CFG
    for a, b in zip(m.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    assert not loaded.training


def test_load_model_dispatches_ckpt_and_dir(tmp_path):
    """Verify loading from both Lightning checkpoints and release directories."""
    from idiom import IDiom

    _, ckpt = _ckpt(tmp_path)
    model_ck, cfg_ck = load_model(ckpt)
    assert cfg_ck == CFG and isinstance(model_ck, IDiomTransformer)

    rel = tmp_path / "rel"
    IDiom.from_lightning_checkpoint(ckpt).save_pretrained(rel)
    model_dir, cfg_dir = load_model(rel)
    assert cfg_dir == CFG
    for a, b in zip(model_ck.state_dict().values(), model_dir.state_dict().values()):
        assert torch.equal(a, b)


def test_lit_modules_persist_model_cfg():
    """Verify that training modules store their model configuration in hyperparameters."""
    from idiom.train.autoreg.lit_autoreg import LitAutoregressive
    from idiom.train.grpo.lit_grpo import LitGRPO

    assert LitAutoregressive(CFG).hparams["model_cfg"] == asdict(CFG)
    terms = lambda idrs, g: ([0.0] * len(idrs), [{}] * len(idrs))  # noqa: E731
    assert LitGRPO(CFG, terms).hparams["model_cfg"] == asdict(CFG)
