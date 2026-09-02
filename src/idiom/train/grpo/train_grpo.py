"""GRPO entrypoint run via idiom_grpo: a pretrained checkpoint plus prompts plus a composite reward.

build_reward_terms composes the reward as a weighted sum of enabled terms (entropy, length, an
RL-SAE feature-code reward, and any number of external reward models), with the tuned defaults in
configs/grpo.yaml. External reward models register via reward.module before launch.
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
from idiom.train.grpo.external import make_external_reward
from idiom.train.grpo.lit_grpo import LitGRPO
from idiom.train.grpo.rewards import (
    entropy_reward, get_reward, length_reward, quadratic_shaping, register_reward, resolve_reward)

# The RL-SAE reward ships in rewards/ (not the installed wheel), registered on import via the same
# reward.module mechanism as any custom reward; the rl_sae config block imports it automatically.
_RL_SAE_MODULE = "rewards/rl_sae.py"


def _register_custom_rewards(spec: str | None) -> None:
    """Import a user module so its register_reward decorators run before reward lookup.

    spec is a dotted module path (for example analysis.my_rewards) or a path to a .py file. The
    module just needs a register_reward("name")-decorated f(idr: str) -> float at import time.

    Args:
        spec (str | None): Dotted module path or .py file path; a no-op when None or empty.
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


def build_reward_terms(rcfg: DictConfig):
    """Build the composite reward: a weighted sum of enabled terms, scored a whole batch at a time.

    The total for each completion is

        total = w_entropy * entropy + w_length * length + w_rl_sae * rl_sae + sum(w_i * external_i)

    Each enabled term contributes weight * score. Terms are scored batch-wise: a plain per-idr
    reward is looped, while a group/batched reward (an external subprocess, an SAE lens) runs once
    for the whole step. A term with monitor=True is logged but left out of the total.

    Legacy configs (a single reward.name, optional reward.group / reward.shaping / reward.monitor)
    are desugared to the same term list first, so old configs and old checkpoints reproduce exactly.

    Args:
        rcfg (DictConfig): Reward config (module plus the entropy, length, rl_sae, and external
            blocks; or the legacy name/group/shaping/monitor/length/entropy shape).

    Returns:
        Callable[[list[str], int], tuple[list[float], list[dict[str, float]]]]: Maps
            (idrs, group_size) to the per-idr totals and a matching per-term breakdown (each dict
            holds every enabled term keyed by its log name, plus total).
    """
    rcfg = _to_new_shape(rcfg)
    _register_custom_rewards(rcfg.get("module"))  # import user rewards before lookup

    # (log_key, weight, scorer(idrs, group_size) -> list[float], monitor)
    terms: list[tuple[str, float, Callable, bool]] = []

    ent = rcfg.get("entropy")
    if ent and ent.get("enabled"):
        te, wd = ent.target_entropy, ent.width
        terms.append(("entropy", float(ent.weight),
                      lambda idrs, gs, te=te, wd=wd:
                          [entropy_reward(x, target_entropy=te, width=wd) for x in idrs], False))

    ln = rcfg.get("length")
    if ln and ln.get("enabled"):
        tl, wd = ln.target_length, ln.width
        terms.append(("length", float(ln.weight),
                      lambda idrs, gs, tl=tl, wd=wd:
                          [length_reward(x, target_length=tl, width=wd) for x in idrs], False))

    rs = rcfg.get("rl_sae")
    if rs and rs.get("enabled"):
        _register_custom_rewards(rs.get("module") or _RL_SAE_MODULE)  # registers sae_only_<sig>
        terms.append(("rl_sae", float(rs.weight), resolve_reward(f"sae_only_{rs.signature}"),
                      bool(rs.get("monitor"))))

    # external rewards ("bring your own"): each entry is EITHER a subprocess scorer (cmd, run in its
    # own environment) OR a registered in-process reward (name, e.g. a custom reward from
    # rewards/builtin_rewards.py). A per-entry module is imported first so its name resolves.
    seen = {k for k, *_ in terms}
    for i, ext in enumerate(rcfg.get("external") or []):
        if not ext.get("enabled"):
            continue
        if ext.get("module"):
            _register_custom_rewards(ext.get("module"))
        label = ext.get("name") or f"external{i}"
        key = label if label not in seen else f"{label}_{i}"
        seen.add(key)
        if ext.get("cmd"):
            scorer = make_external_reward(
                ext.cmd, target=ext.get("target"), width=float(ext.get("width", 1.0)),
                timeout=float(ext.get("timeout", 300.0)), maxlen=int(ext.get("maxlen", 0)), label=key)
        else:
            scorer = resolve_reward(ext.name)
        terms.append((key, float(ext.weight), scorer, bool(ext.get("monitor"))))

    def score_batch(idrs: list[str], group_size: int):
        n = len(idrs)
        totals = [0.0] * n
        breakdown: list[dict[str, float]] = [{} for _ in range(n)]
        for key, weight, scorer, monitor in terms:
            scores = scorer(idrs, group_size)
            for i in range(n):
                if monitor:
                    breakdown[i][key] = scores[i]          # logged, excluded from total
                else:
                    val = weight * scores[i]
                    breakdown[i][key] = val
                    totals[i] += val
        for i in range(n):
            breakdown[i]["total"] = totals[i]
        return totals, breakdown

    return score_batch


def _to_new_shape(rcfg: DictConfig) -> DictConfig:
    """Desugar a legacy reward config (name/group/shaping/monitor) into the term-block shape.

    A config that already has an rl_sae or external block is returned unchanged. Otherwise the
    legacy base reward (group takes precedence over name, matching the old build) becomes a single
    external term of weight 1.0; a legacy monitor becomes a monitor term; and legacy shaping is
    preserved by wrapping the base reward in a registered shaped alias. entropy and length blocks
    carry over untouched.

    Args:
        rcfg (DictConfig): A reward config in either shape.

    Returns:
        DictConfig: The config in the new term-block shape.
    """
    if "rl_sae" in rcfg or "external" in rcfg:
        return rcfg
    d = OmegaConf.to_container(rcfg, resolve=True)
    external = []
    base = d.get("group") or d.get("name")  # legacy: group overrode name
    if base:
        _register_custom_rewards(d.get("module"))  # so the base name resolves before we wrap it
        name = base
        sh = d.get("shaping") or {}
        if sh.get("enabled"):
            fn = get_reward(base)
            name = f"__shaped__{base}"
            register_reward(name)(
                lambda idr, fn=fn, t=sh["target"], s=sh["scale"]:
                    quadratic_shaping(fn(idr), target=t, scale=s))
        external.append({"enabled": True, "weight": 1.0, "name": name})
    if d.get("monitor"):
        external.append({"enabled": True, "weight": 0.0, "name": d["monitor"], "monitor": True})
    for k in ("name", "group", "shaping", "monitor"):
        d.pop(k, None)
    d["external"] = external
    return OmegaConf.create(d)


def build_reward_components(rcfg: DictConfig) -> Callable[[str], dict[str, float]]:
    """Per-idr reward breakdown (back-compat shim over build_reward_terms).

    Args:
        rcfg (DictConfig): Reward config in either shape.

    Returns:
        Callable[[str], dict[str, float]]: Maps an IDR to its per-term breakdown (including total).
    """
    terms = build_reward_terms(rcfg)
    return lambda idr: terms([idr], 1)[1][0]


def build_reward(rcfg: DictConfig) -> Callable[[str], float]:
    """Scalar reward the policy optimizes (back-compat shim over build_reward_terms).

    Args:
        rcfg (DictConfig): Reward config in either shape.

    Returns:
        Callable[[str], float]: Maps an IDR to its scalar total reward.
    """
    terms = build_reward_terms(rcfg)
    return lambda idr: terms([idr], 1)[0][0]


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
