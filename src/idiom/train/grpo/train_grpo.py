"""GRPO entrypoint run via idiom_grpo: a pretrained checkpoint plus prompts plus a composite reward.

build_reward_terms composes the reward as a weighted sum of enabled terms (entropy, length, an
RL-SAE feature-code reward, and any number of external reward models), with the tuned defaults in
configs/grpo.yaml. External reward models register via reward.module before launch.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from idiom.train.grpo.data import collate_prompts, idp_prompts, record_prompts
from idiom.train.grpo.lit_grpo import LitGRPO

# The composite reward lives in reward/ (build_reward/_components are back-compat shims over
# build_reward_terms, re-exported here so old imports from train_grpo keep working).
from idiom.train.grpo.reward import build_reward, build_reward_components, build_reward_terms

__all__ = ["build", "build_reward", "build_reward_components", "build_reward_terms", "run"]


def build(cfg: DictConfig) -> tuple[LitGRPO, object]:
    """Wire the GRPO module and prompt dataset from a resolved config.

    Always warm-starts the policy from a pretrained checkpoint (cfg.init_from); the architecture
    is read from that checkpoint. Builds the composite reward from cfg.reward, and an unprompted or
    prompted-IDR prompt set based on cfg.prompts.mode (the legacy modes idp and denovo are accepted
    for unprompted).

    Args:
        cfg (DictConfig): Resolved GRPO config (grpo, reward, prompts, init_from).

    Returns:
        tuple[LitGRPO, object]: The GRPO module and its prompt dataset.
    """
    grpo_kw = OmegaConf.to_container(cfg.grpo, resolve=True)
    # GRPO always warm-starts from a pretrained policy; architecture is read from that checkpoint.
    # One composite reward: a weighted sum of the enabled terms, scored a whole batch per step.
    reward_terms = build_reward_terms(cfg.reward)
    lit = LitGRPO.init_from_checkpoint(cfg.init_from, reward_terms=reward_terms, **grpo_kw)

    # "idp"/"denovo" kept for back-compat with old configs; "unprompted" is the current vocabulary
    if cfg.prompts.mode in ("unprompted", "idp", "denovo"):
        ds = idp_prompts(cfg.prompts.n)
    else:
        ds = record_prompts(cfg.prompts.fasta, cfg.prompts.n_per)
    return lit, ds


def run(cfg: DictConfig) -> None:
    """Build the module and prompts, configure the trainer and logger, and fit.

    Args:
        cfg (DictConfig): Resolved GRPO config.
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
    # Plot every metric against the REAL training step. Lightning logs trainer/global_step as a
    # metric but lets W&B's internal _step increment once per log flush (= every log_every_n_steps
    # steps), which compresses the default x-axis; this makes global_step the x-axis so a step is a step.
    try:
        wandb_logger.experiment.define_metric("trainer/global_step")
        wandb_logger.experiment.define_metric("*", step_metric="trainer/global_step")
    except Exception:  # noqa: BLE001 - offline/disabled W&B has no experiment to configure
        pass
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    ckpt_every = trainer_cfg.pop("checkpoint_every", 0)
    max_steps = trainer_cfg.get("max_steps") or None
    # Step-based checkpointing only -- NO per-epoch saves (setting every_n_train_steps makes Lightning
    # skip epoch-end saves). checkpoint_every>0 -> keep every N steps + last.ckpt; checkpoint_every=0
    # -> save ONLY the final-step checkpoint (step_step=<max_steps>.ckpt), nothing else.
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
    """Hydra entrypoint that runs GRPO post-training with the composed config."""
    run(cfg)


if __name__ == "__main__":
    main()
