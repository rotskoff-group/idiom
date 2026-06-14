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
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

from idiom.data.datamodule import RecordDataModule
from idiom.model.config import ModelConfig
from idiom.train.lit_autoregressive import LitAutoregressive


def build(cfg: DictConfig) -> tuple[LitAutoregressive, RecordDataModule]:
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    optim = OmegaConf.to_container(cfg.optim, resolve=True)
    optim["max_steps"] = cfg.trainer.max_steps  # scheduler shares the trainer's horizon

    if cfg.get("init_from"):
        lit = LitAutoregressive.init_from_checkpoint(cfg.init_from, model_cfg, **optim)
        log.info(f"Warm-started from {cfg.init_from}")
    else:
        lit = LitAutoregressive(model_cfg, **optim)

    dm = RecordDataModule(
        cfg.data.train_fasta,
        cfg.data.get("val_fasta"),
        max_len=model_cfg.max_seq_len,
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
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=[ModelCheckpoint(dirpath=out_dir / "checkpoints", save_last=True)],
        default_root_dir=out_dir,
    )
    trainer.fit(lit, datamodule=dm)


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
