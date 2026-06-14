"""P3 training tests (CPU-only): masked loss, SFT completion mask, warmup-cosine, fit smoke."""

import lightning as L
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
    ds = RecordDataset(RECS, TOK, max_len=64, fim_full_prob=1.0, completion_only=completion_only)
    return DataLoader(ds, batch_size=2, collate_fn=make_collate(TOK.pad_id))


def test_sft_mask_is_completion_only():
    # SFT: only the IDR (idr_len residues) + STOP carry loss.
    ds = RecordDataset(RECS, TOK, fim_full_prob=1.0, completion_only=True)
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
    assert lrs[0] < lrs[4] and abs(max(lrs) - 1.0) < 1e-6  # warms up to base LR
    assert lrs[-1] < lrs[5] and lrs[-1] >= 0.1 - 1e-6  # decays toward min_lr_ratio


def test_init_from_checkpoint_roundtrip(tmp_path):
    # Save a pretrained module, warm-start a new one, and confirm weights transfer (SFT path).
    lit = LitAutoregressive(TINY)
    ckpt = tmp_path / "pre.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in lit.model.state_dict().items()}}, ckpt)
    sft = LitAutoregressive.init_from_checkpoint(str(ckpt), TINY, lr=1e-5)
    for (k, a), (_, b) in zip(lit.model.state_dict().items(), sft.model.state_dict().items()):
        assert torch.equal(a, b), k


def test_build_wires_pretrain_and_sft(tmp_path):
    from omegaconf import OmegaConf

    from idiom.train.train import build

    fasta = tmp_path / "train.fasta"
    fasta.write_text(">A_IDR_3-6\nMEDSKVDNRPQ\n>B_IDR_2-5\nACDEFGHIKL\n")
    base = {
        "seed": 0,
        "model": {"n_layers": 2, "d_model": 32, "n_heads": 4, "max_seq_len": 64, "vocab_size": 27},
        "optim": {"lr": 3e-4, "warmup_steps": 1, "weight_decay": 0.1, "min_lr_ratio": 0.1},
        "trainer": {"max_steps": 5},
        "data": {"train_fasta": str(fasta), "val_fasta": None, "fim_full_prob": 1.0,
                 "completion_only": True, "batch_size": 2, "num_workers": 0},
    }
    lit, dm = build(OmegaConf.create(base))
    assert lit.model.cfg.n_layers == 2 and lit.max_steps_ == 5
    assert dm.completion_only is True  # SFT-style data wiring propagated


def test_trainer_fit_smoke(tmp_path):
    # exercises configure_optimizers + scheduler + a couple of optimizer steps, CPU-only.
    lit = LitAutoregressive(TINY, warmup_steps=1, max_steps=2)
    trainer = L.Trainer(
        max_steps=2, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(lit, train_dataloaders=_loader())
    assert trainer.global_step == 2
