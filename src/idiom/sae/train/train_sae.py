"""Hydra entrypoint for SAE training and release export."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.fim import PROMPTED, UNPROMPTED
from idiom.data.record_store import open_or_build
from idiom.data.tokenizer import Tokenizer
from idiom.model.io import load_model
from idiom.sae.model.io import save_sae
from idiom.sae.train.activation_store import ActivationStore
from idiom.sae.train.lit_sae import LitSAE
from idiom.utils.device import resolve_device


def build(cfg: DictConfig) -> tuple[LitSAE, ActivationStore]:
    """Wire the host model, record source, activation store, and LitSAE from a config.

    The host model is loaded from cfg.model_ckpt and its d_model sets the SAE input width.

    Args:
        cfg: Resolved SAE training config.

    Returns:
        The LightningModule and its activation store.
    """
    device = resolve_device(cfg.device)
    tok = Tokenizer()
    model, model_cfg = load_model(cfg.model_ckpt, device=device)

    records = RecordDataset(
        open_or_build(cfg.data.fasta), tok, max_len=model_cfg.max_seq_len,
        prompted_prob=cfg.data.get("prompted_prob", 0.5),
    )
    record_loader = DataLoader(
        records, batch_size=cfg.data.record_batch_size, collate_fn=make_collate(tok.pad_id),
        # Shuffle to avoid training only on the file head when max_steps is finite
        shuffle=cfg.data.get("shuffle", True),
    )
    store = ActivationStore(
        model, record_loader, cfg.layer, sae_batch_size=cfg.sae_batch_size,
        buffer_size=cfg.buffer_size, device=device, tokenizer=tok,
        region=cfg.get("region", "all"),
    )
    lit = LitSAE(
        d_in=model_cfg.d_model,
        k=cfg.sae.k,
        expansion_factor=cfg.sae.expansion_factor,
        activation=cfg.sae.activation,
        multi_topk=cfg.sae.multi_topk,
        auxk_alpha=cfg.sae.auxk_alpha,
        dead_feature_tokens=cfg.sae.dead_feature_tokens,
        total_steps=cfg.trainer.max_steps,
        warmup_steps=cfg.sae.warmup_steps,
    )
    return lit, store


def run(cfg: DictConfig) -> None:
    """Fit the SAE and write the release directory.

    Writes the resolved config to cfg.out_dir, logs to Weights & Biases, and on completion saves
    sae_config.json and sae.safetensors recording the host model, layer, region, and prompt format.
    The recorded fim_mode is "unprompted" only when cfg.data.prompted_prob is 0.

    Args:
        cfg: Resolved SAE training config.
    """
    L.seed_everything(cfg.seed, workers=True)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    lit, store = build(cfg)
    if cfg.init_b_dec_from_mean:
        lit.init_b_dec_from_mean(store.mean_activation())
    dl = DataLoader(store, batch_size=None)  # the store already yields [B, d_model] batches
    wandb_logger = WandbLogger(
        project=cfg.get("wandb_project", "idiom-sae"), name=cfg.get("run_name"), save_dir=str(out_dir)
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True), logger=wandb_logger, default_root_dir=out_dir
    )
    trainer.fit(lit, train_dataloaders=dl, ckpt_path=cfg.get("resume_from"))
    # Any mixture with flanking context is recorded as prompted
    fim_mode = UNPROMPTED if float(cfg.data.get("prompted_prob", 0.5)) == 0.0 else PROMPTED
    save_sae(
        lit.sae, out_dir, host_model=str(cfg.model_ckpt), layer=cfg.layer,
        region=cfg.get("region", "all"), fim_mode=fim_mode,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sae")
def main(cfg: DictConfig) -> None:
    """Train an SAE from the Hydra config."""
    run(cfg)


if __name__ == "__main__":
    main()
