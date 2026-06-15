"""GRPO entrypoint (``idiom_grpo``). Pretrained ckpt + prompts + a composite reward -> RL.

``build_reward`` composes a base reward with optional quadratic shaping + length + entropy
terms (the sweep-tuned defaults live in ``configs/grpo.yaml``). ProtGPS is operator-wired (it
loads a vendored model) — register it under the name ``protgps`` before launching.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from idiom.model.config import ModelConfig
from idiom.train.grpo.data import collate_prompts, denovo_prompts, record_prompts
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.rewards import entropy_reward, get_reward, length_reward, quadratic_shaping


def _register_custom_rewards(spec: str | None) -> None:
    """Import a user module so its ``@register_reward`` decorators run before reward lookup.

    ``spec`` is a dotted module path (e.g. ``analysis.my_rewards``) or a ``*.py`` file path. The
    module just needs ``@register_reward("name") def f(idr: str) -> float: ...`` at import time.
    """
    if not spec:
        return
    import importlib
    import importlib.util

    if spec.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location("idiom_custom_rewards", spec)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        importlib.import_module(spec)


def build_reward(rcfg: DictConfig) -> Callable[[str], float]:
    _register_custom_rewards(rcfg.get("module"))  # user rewards (or operator-registered protgps)
    base = get_reward(rcfg.name)

    def reward(idr: str) -> float:
        r = base(idr)
        if rcfg.shaping.enabled:
            r = quadratic_shaping(r, target=rcfg.shaping.target, scale=rcfg.shaping.scale)
        if rcfg.length.enabled:
            r += rcfg.length.weight * length_reward(
                idr, target_length=rcfg.length.target_length, width=rcfg.length.width
            )
        if rcfg.entropy.enabled:
            r += rcfg.entropy.weight * entropy_reward(
                idr, target_entropy=rcfg.entropy.target_entropy, width=rcfg.entropy.width
            )
        return r

    return reward


def build(cfg: DictConfig) -> tuple[LitGRPO, object]:
    reward = build_reward(cfg.reward)
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    grpo_kw = OmegaConf.to_container(cfg.grpo, resolve=True)

    if cfg.get("init_from"):
        lit = LitGRPO.init_from_checkpoint(cfg.init_from, model_cfg, reward, **grpo_kw)
    else:
        lit = LitGRPO(model_cfg, reward, **grpo_kw)

    if cfg.prompts.mode == "denovo":
        ds = denovo_prompts(cfg.prompts.n)
    else:
        ds = record_prompts(cfg.prompts.fasta, cfg.prompts.n_per)
    return lit, ds


def run(cfg: DictConfig) -> None:
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
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=[ModelCheckpoint(dirpath=out_dir / "checkpoints", save_last=True)],
        logger=wandb_logger,
        default_root_dir=out_dir,
    )
    trainer.fit(lit, train_dataloaders=dl, ckpt_path=cfg.get("resume_from"))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="grpo")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
