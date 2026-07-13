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

from idiom.train.grpo.data import collate_prompts, idp_prompts, record_prompts
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.rewards import (
    entropy_reward, get_group_reward, get_reward, length_reward, quadratic_shaping)


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


def build_reward_components(rcfg: DictConfig) -> Callable[[str], dict[str, float]]:
    """Composed reward that returns a per-term breakdown for logging.

    Always includes ``raw`` (the base reward before any shaping) and ``total`` (the scalar the
    policy optimizes); ``length``/``entropy`` appear only when those terms are enabled. ``raw`` is
    what gets logged as ``train/reward_raw`` so the base signal is visible separately from shaping.
    """
    _register_custom_rewards(rcfg.get("module"))  # user rewards (or operator-registered protgps)
    base = get_reward(rcfg.name)
    # optional monitor reward: computed + logged (train/reward_monitor) but NOT added to total, so
    # you can watch a metric (e.g. ProtGPS) during a run that does not optimize it.
    monitor_name = rcfg.get("monitor")
    monitor = get_reward(monitor_name) if monitor_name else None

    def components(idr: str) -> dict[str, float]:
        raw = base(idr)
        total = raw
        if rcfg.shaping.enabled:
            total = quadratic_shaping(total, target=rcfg.shaping.target, scale=rcfg.shaping.scale)
        out = {"raw": raw}
        if rcfg.length.enabled:
            lr = rcfg.length.weight * length_reward(
                idr, target_length=rcfg.length.target_length, width=rcfg.length.width
            )
            out["length"] = lr
            total += lr
        if rcfg.entropy.enabled:
            er = rcfg.entropy.weight * entropy_reward(
                idr, target_entropy=rcfg.entropy.target_entropy, width=rcfg.entropy.width
            )
            out["entropy"] = er
            total += er
        if monitor is not None:
            out["monitor"] = monitor(idr)   # logged only; deliberately excluded from `total`
        out["total"] = total
        return out

    return components


def build_reward(rcfg: DictConfig) -> Callable[[str], float]:
    """Scalar reward the policy optimizes (the ``total`` term of :func:`build_reward_components`)."""
    components = build_reward_components(rcfg)
    return lambda idr: components(idr)["total"]


def build_group_reward_components(rcfg: DictConfig):
    """Group-aware breakdown: f(idrs, group_size) -> list of per-idr {raw, length, entropy, monitor,
    total}. `raw` is a GROUP reward (each completion scored relative to its GRPO group, e.g. SAE-code
    coverage); length/entropy/monitor stay per-idr, mirroring build_reward_components. `total` excludes
    `monitor`. Shaping is not applied (group rewards are already relative)."""
    _register_custom_rewards(rcfg.get("module"))
    group_base = get_group_reward(rcfg.group)
    monitor_name = rcfg.get("monitor")
    monitor = get_reward(monitor_name) if monitor_name else None

    def components(idrs, group_size):
        base = group_base(idrs, group_size)
        out = []
        for idr, b in zip(idrs, base):
            total = b
            d = {"raw": b}
            if rcfg.length.enabled:
                lr = rcfg.length.weight * length_reward(
                    idr, target_length=rcfg.length.target_length, width=rcfg.length.width)
                d["length"] = lr
                total += lr
            if rcfg.entropy.enabled:
                er = rcfg.entropy.weight * entropy_reward(
                    idr, target_entropy=rcfg.entropy.target_entropy, width=rcfg.entropy.width)
                d["entropy"] = er
                total += er
            if monitor is not None:
                d["monitor"] = monitor(idr)
            d["total"] = total
            out.append(d)
        return out

    return components


def build(cfg: DictConfig) -> tuple[LitGRPO, object]:
    grpo_kw = OmegaConf.to_container(cfg.grpo, resolve=True)
    # GRPO always warm-starts from a pretrained policy; architecture is read from that checkpoint.
    if cfg.reward.get("group"):  # group-aware reward (population coverage): scored per GRPO group
        gcomp = build_group_reward_components(cfg.reward)
        lit = LitGRPO.init_from_checkpoint(
            cfg.init_from, lambda idr: 0.0, group_reward_components=gcomp, **grpo_kw
        )
    else:  # standard per-idr reward + per-term logging (train/reward_raw, _length, _entropy)
        components = build_reward_components(cfg.reward)
        reward = lambda idr: components(idr)["total"]  # noqa: E731 - scalar the policy optimizes
        lit = LitGRPO.init_from_checkpoint(
            cfg.init_from, reward, reward_components=components, **grpo_kw
        )

    if cfg.prompts.mode in ("idp", "denovo"):  # "denovo" kept for back-compat with old configs
        ds = idp_prompts(cfg.prompts.n)
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
    # Plot every metric against the REAL training step. Lightning logs `trainer/global_step` as a
    # metric but lets W&B's internal `_step` increment once per log flush (= every log_every_n_steps
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
    run(cfg)


if __name__ == "__main__":
    main()
