"""Hydra entrypoint for pretraining and SFT (--config-name sft)."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

from idiom.data.datamodule import RecordDataModule
from idiom.model.config import ModelConfig
from idiom.train.autoreg.lit_autoreg import LitAutoregressive


def build(cfg: DictConfig) -> tuple[LitAutoregressive, RecordDataModule]:
    """Wire the training module and datamodule from a resolved config.

    With cfg.init_from set the module is warm-started and its architecture comes from that
    artifact; otherwise the architecture is built from cfg.model. The scheduler horizon is
    cfg.trainer.max_steps.

    Args:
        cfg: Resolved training config, with optim, trainer, data, seed, and either model or
            init_from.

    Returns:
        The training module and its datamodule.
    """
    optim = OmegaConf.to_container(cfg.optim, resolve=True)
    optim["max_steps"] = cfg.trainer.max_steps

    if cfg.get("init_from"):
        lit = LitAutoregressive.init_from_checkpoint(cfg.init_from, **optim)
        log.info(f"Warm-started from {cfg.init_from}")
    else:
        lit = LitAutoregressive(ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True)), **optim)

    dm = RecordDataModule(
        cfg.data.train_fasta,
        cfg.data.get("val_fasta"),
        max_len=lit.cfg.max_seq_len,
        prompted_prob=cfg.data.get("prompted_prob", 0.5),
        completion_only=cfg.data.completion_only,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )
    return lit, dm


def run(cfg: DictConfig) -> None:
    """Build the module and data, configure the trainer and logger, and fit.

    Save the resolved config and checkpoints under cfg.out_dir and log to W&B.
    Keep a rolling last.ckpt every cfg.ckpt_every_n_steps and the three best validation
    checkpoints. Resume training state from cfg.resume_from when set.

    Args:
        cfg: Resolved training config.
    """
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
    has_val = bool(cfg.data.get("val_fasta"))
    if not has_val:
        trainer_kw["limit_val_batches"] = 0
        trainer_kw["num_sanity_val_steps"] = 0
    # Only the rolling callback writes last.ckpt; validation saves the three best models
    ckpt_dir = out_dir / "checkpoints"
    every_n = int(cfg.get("ckpt_every_n_steps", 2000))
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=0, save_last=True, every_n_train_steps=every_n),
        LearningRateMonitor(logging_interval="step"),
    ]
    if has_val:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                filename="epoch_{epoch}_step_{step}",
                auto_insert_metric_name=False,
            ),
        )
    # Spawn local ranks for single-node jobs; use launcher-provided ranks for multi-node jobs
    plugins = [LightningEnvironment()] if trainer_kw.get("num_nodes", 1) == 1 else None
    trainer = L.Trainer(
        **trainer_kw,
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=out_dir,
        plugins=plugins,
    )
    trainer.fit(lit, datamodule=dm, ckpt_path=cfg.get("resume_from"))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    """Run pretraining or SFT from the Hydra config."""
    run(cfg)


if __name__ == "__main__":
    main()
