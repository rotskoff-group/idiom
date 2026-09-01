"""SAE training entrypoint (idiom_sae): frozen IDiom -> streaming activations -> top-k SAE.

Activations are generated on the fly by the ActivationStore (no activation cache on disk). build
wires the frozen model + record source + store + LitSAE; run fits and writes the release directory
(sae_config.json + sae.safetensors, with host_model / layer / region recorded).
"""

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
from idiom.model.io import load_pretrained
from idiom.sae.io import save_sae
from idiom.sae.training.activation_store import ActivationStore
from idiom.sae.training.lit_sae import LitSAE
from idiom.utils.device import resolve_device


def _prompted_prob(cfg: DictConfig) -> float:
    """Return the probability of the prompted variant from config.

    Accepts the deprecated fim_idr_prob / fim_full_prob keys as fallbacks.

    Args:
        cfg (DictConfig): The training config.

    Returns:
        float: The prompted-variant probability.
    """
    return cfg.data.get(
        "prompted_prob", cfg.data.get("fim_idr_prob", cfg.data.get("fim_full_prob", 0.5))
    )


def build(cfg: DictConfig) -> tuple[LitSAE, ActivationStore]:
    """Wire the frozen model, record source, activation store, and LitSAE from config.

    Args:
        cfg (DictConfig): The training config.

    Returns:
        tuple[LitSAE, ActivationStore]: The LightningModule and its streaming activation store.
    """
    device = resolve_device(cfg.device)
    tok = Tokenizer()
    model = load_pretrained(cfg.model_ckpt, device=device)  # arch read from the checkpoint
    model_cfg = model.cfg

    records = RecordDataset(
        open_or_build(cfg.data.fasta), tok, max_len=model_cfg.max_seq_len,
        prompted_prob=_prompted_prob(cfg),
    )
    record_loader = DataLoader(
        records, batch_size=cfg.data.record_batch_size, collate_fn=make_collate(tok.pad_id),
        shuffle=cfg.data.get("shuffle", True),  # random record order (RecordDataset is map-style);
        # with a finite max_steps this makes the consumed slice a uniform draw over the whole corpus
        # rather than the file head. Default on; set data.shuffle=false to restore file-order streaming.
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
    """Fit the SAE and write the release directory recording host_model / layer / region / fim_mode.

    Args:
        cfg (DictConfig): The training config.
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
    # canonical SAE release (host_model + layer + region + fim_mode recorded): loads via
    # IDiomSAE.from_pretrained. fim_mode is "unprompted" iff training was pure de-novo
    # (prompted_prob==0), else "prompted" — the prompt format downstream tools rebuild activations under.
    fim_mode = UNPROMPTED if float(_prompted_prob(cfg)) == 0.0 else PROMPTED
    save_sae(
        lit.sae, out_dir, host_model=str(cfg.model_ckpt), layer=cfg.layer,
        region=cfg.get("region", "all"), fim_mode=fim_mode,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sae")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
