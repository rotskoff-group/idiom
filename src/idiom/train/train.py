"""Training entrypoint (pretrain + SFT). ``idiom_train`` / ``idiom_train --config-name sft``.

``build(cfg)`` wires model + data + module (and is unit-testable); ``run(cfg)`` fits. Pretrain
vs SFT is entirely in the config: SFT sets ``init_from`` (warm start) and
``data.completion_only=true`` (loss on the IDR only).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

from idiom.data.datamodule import RecordDataModule
from idiom.model.config import ModelConfig
from idiom.train.lit_autoregressive import LitAutoregressive


def build(cfg: DictConfig) -> tuple[LitAutoregressive, RecordDataModule]:
    optim = OmegaConf.to_container(cfg.optim, resolve=True)
    optim["max_steps"] = cfg.trainer.max_steps  # scheduler shares the trainer's horizon

    if cfg.get("init_from"):
        # SFT: warm start; architecture comes from the pretrained checkpoint, not the config
        lit = LitAutoregressive.init_from_checkpoint(cfg.init_from, **optim)
        log.info(f"Warm-started from {cfg.init_from}")
    else:
        # pretraining: this is where the architecture is defined
        lit = LitAutoregressive(ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True)), **optim)

    dm = RecordDataModule(
        cfg.data.train_fasta,
        cfg.data.get("val_fasta"),
        max_len=lit.cfg.max_seq_len,
        fim_full_prob=cfg.data.fim_full_prob,
        completion_only=cfg.data.completion_only,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )
    return lit, dm


def run(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    lit, dm = build(cfg)
    wandb_logger = WandbLogger(
        project=cfg.get("wandb_project", "idiom"), name=cfg.get("run_name"), save_dir=str(out_dir)
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer_kw = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not cfg.data.get("val_fasta"):  # no held-out set (e.g. SFT) -> turn the val loop off entirely
        trainer_kw["limit_val_batches"] = 0
        trainer_kw["num_sanity_val_steps"] = 0
    trainer = L.Trainer(
        **trainer_kw,
        callbacks=[ModelCheckpoint(dirpath=out_dir / "checkpoints", save_last=True)],
        logger=wandb_logger,
        default_root_dir=out_dir,
    )
    # resume_from restores optimizer state + global step + LR schedule + RNG (a true mid-run resume,
    # e.g. after preemption); distinct from init_from, which is a weights-only warm start. Mutually
    # exclusive — set at most one.
    trainer.fit(lit, datamodule=dm, ckpt_path=cfg.get("resume_from"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
