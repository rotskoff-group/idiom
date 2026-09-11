"""Hydra entrypoint for GRPO with explicitly configured reward terms."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from idiom.data.fim import UNPROMPTED, normalize_mode
from idiom.train.grpo.data import collate_prompts, prompted_prompts, unprompted_prompts
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.reward import build_reward

__all__ = ["build", "build_reward", "run"]


def build(cfg: DictConfig) -> tuple[LitGRPO, object]:
    """Wire the GRPO module and prompt dataset from a resolved config.

    Load weights and architecture from cfg.init_from. Unprompted mode repeats "132"
    cfg.prompts.n times; prompted mode repeats each FASTA record's flank prompt
    cfg.prompts.n_per times.

    Args:
        cfg: Resolved GRPO config, with grpo, reward, prompts, and init_from.

    Returns:
        The GRPO module and its prompt dataset.

    Raises:
        ValueError: If cfg.prompts.mode is neither "unprompted" nor "prompted".
    """
    grpo_kw = OmegaConf.to_container(cfg.grpo, resolve=True)
    reward_terms = build_reward(cfg.reward)
    lit = LitGRPO.init_from_checkpoint(cfg.init_from, reward_terms=reward_terms, **grpo_kw)

    if normalize_mode(cfg.prompts.mode) == UNPROMPTED:
        ds = unprompted_prompts(cfg.prompts.n)
    else:
        ds = prompted_prompts(cfg.prompts.fasta, cfg.prompts.n_per)
    return lit, ds


def run(cfg: DictConfig) -> None:
    """Build the module and prompts, configure the trainer and logger, and fit.

    Writes the resolved config to cfg.out_dir and logs to Weights & Biases against the training
    step. With cfg.trainer.checkpoint_every above 0, a checkpoint is kept every that many steps
    alongside last.ckpt; otherwise only the final step is saved.

    Args:
        cfg: Resolved GRPO config.
    """
    L.seed_everything(cfg.seed, workers=True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    lit, ds = build(cfg)
    dl = DataLoader(ds, batch_size=cfg.prompts.batch_size, shuffle=True, collate_fn=collate_prompts)
    wandb_logger = WandbLogger(
        project=cfg.get("wandb_project", "idiom-grpo"), name=cfg.get("run_name"), save_dir=str(out_dir)
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    # Use training steps for the W&B x-axis; its default step counts log flushes
    try:
        wandb_logger.experiment.define_metric("trainer/global_step")
        wandb_logger.experiment.define_metric("*", step_metric="trainer/global_step")
    except Exception: # offline/disabled W&B has no experiment to configure
        pass
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    ckpt_every = trainer_cfg.pop("checkpoint_every", 0)
    max_steps = trainer_cfg.get("max_steps") or None
    # Disable epoch-end saves; checkpoint_every=0 saves only the final step
    callbacks = [ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        every_n_train_steps=ckpt_every or max_steps,
        save_top_k=-1 if ckpt_every else 1,
        save_last=bool(ckpt_every),
        filename="step_{step}",
    )]
    trainer = L.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=out_dir,
    )
    trainer.fit(lit, train_dataloaders=dl, ckpt_path=cfg.get("resume_from"))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="grpo")
def main(cfg: DictConfig) -> None:
    """Run GRPO post-training from the Hydra config."""
    run(cfg)


if __name__ == "__main__":
    main()
