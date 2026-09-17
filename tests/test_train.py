"""Training tests: masked loss, SFT completion mask, warmup-cosine, fit smoke."""

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.io import Record
from idiom.data.tokenizer import Tokenizer
from idiom.model import ModelConfig
from idiom.train import LitAutoregressive, warmup_cosine

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=32, n_heads=4, max_seq_len=64)
RECS = [Record("a", "MEDSKVDNRPQ", 2, 6), Record("b", "ACDEFGHIKLWY", 3, 9)]


def _loader(completion_only=False):
    ds = RecordDataset(RECS, TOK, max_len=64, prompted_prob=1.0, completion_only=completion_only)
    return DataLoader(ds, batch_size=2, collate_fn=make_collate(TOK.pad_id))


def test_sft_mask_is_completion_only():
    # SFT: only the IDR (idr_len residues) + STOP carry loss
    ds = RecordDataset(RECS, TOK, prompted_prob=1.0, completion_only=True)
    _, y, mask = ds[0]
    idr_len = RECS[0].idr_end - RECS[0].idr_start
    assert int(mask.sum()) == idr_len + 1
    assert mask[-(idr_len + 1) :].all() and not mask[: -(idr_len + 1)].any()


def test_training_step_finite_grad():
    lit = LitAutoregressive(TINY, warmup_steps=1, max_steps=10)
    batch = next(iter(_loader()))
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss) and loss.requires_grad


def test_warmup_cosine_shape():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=5, max_steps=20, min_lr_ratio=0.1)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[4] and abs(max(lrs) - 1.0) < 1e-6
    assert lrs[-1] < lrs[5] and lrs[-1] >= 0.1 - 1e-6


def test_init_from_checkpoint_roundtrip(tmp_path):
    lit = LitAutoregressive(TINY)
    ckpt = tmp_path / "pre.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in lit.model.state_dict().items()},
            "hyper_parameters": dict(lit.hparams),
        },
        ckpt,
    )
    sft = LitAutoregressive.init_from_checkpoint(str(ckpt), lr=1e-5)
    assert sft.cfg == TINY
    for (k, a), (_, b) in zip(lit.model.state_dict().items(), sft.model.state_dict().items()):
        assert torch.equal(a, b), k


def test_build_wires_pretrain_and_sft(tmp_path):
    from omegaconf import OmegaConf

    from idiom.train.autoreg.train_autoreg import build

    fasta = tmp_path / "train.fasta"
    fasta.write_text(">A_IDR_3-6\nMEDSKVDNRPQ\n>B_IDR_2-5\nACDEFGHIKL\n")
    base = {
        "seed": 0,
        "model": {"n_layers": 2, "d_model": 32, "n_heads": 4, "max_seq_len": 64, "vocab_size": 27},
        "optim": {"lr": 3e-4, "warmup_steps": 1, "weight_decay": 0.1, "min_lr_ratio": 0.1},
        "trainer": {"max_steps": 5},
        "data": {
            "train_fasta": str(fasta),
            "val_fasta": None,
            "prompted_prob": 1.0,
            "completion_only": True,
            "batch_size": 2,
            "num_workers": 0,
        },
    }
    lit, dm = build(OmegaConf.create(base))
    assert lit.model.cfg.n_layers == 2 and lit.max_steps_ == 5
    assert dm.completion_only is True


def test_trainer_fit_smoke(tmp_path):
    lit = LitAutoregressive(TINY, warmup_steps=1, max_steps=2)
    trainer = L.Trainer(
        max_steps=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(lit, train_dataloaders=_loader())
    assert trainer.global_step == 2


@pytest.mark.parametrize("nodes", [1, 2])
def test_autoreg_runner_preserves_external_launcher_for_multiple_nodes(tmp_path, monkeypatch, nodes):
    from types import SimpleNamespace

    from lightning.pytorch.plugins.environments import LightningEnvironment
    from omegaconf import OmegaConf

    from idiom.train.autoreg import train_autoreg

    captured = {}
    module, data = object(), object()
    monkeypatch.setattr(train_autoreg, "build", lambda cfg: (module, data))
    monkeypatch.setattr(
        train_autoreg, "WandbLogger", lambda **kw: SimpleNamespace(log_hyperparams=lambda cfg: None)
    )

    def trainer(**kw):
        captured.update(kw)
        return SimpleNamespace(fit=lambda lit, **args: captured.update(lit=lit, fit_args=args))

    monkeypatch.setattr(train_autoreg.L, "Trainer", trainer)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "out_dir": str(tmp_path),
            "data": {"val_fasta": None},
            "trainer": {"num_nodes": nodes, "devices": 4},
        }
    )
    train_autoreg.run(cfg)
    if nodes == 1:
        assert len(captured["plugins"]) == 1
        assert isinstance(captured["plugins"][0], LightningEnvironment)
    else:
        assert captured["plugins"] is None
    assert captured["num_nodes"] == nodes and captured["devices"] == 4
    assert captured["lit"] is module and captured["fit_args"]["datamodule"] is data
