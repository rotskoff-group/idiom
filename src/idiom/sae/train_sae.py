"""SAE training entrypoint (``idiom_sae``): frozen IDiom -> streaming activations -> top-k SAE.

Activations are generated on the fly by the :class:`ActivationStore` (no h5, D3). ``build``
wires the frozen model + record source + store + ``LitSAE``; ``run`` fits and saves ``ae.pt``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.record_store import open_or_build
from idiom.data.tokenizer import Tokenizer
from idiom.model.io import load_pretrained
from idiom.sae.activation_store import ActivationStore
from idiom.sae.io import save_sae
from idiom.sae.lit_sae import LitSAE
from idiom.utils.device import resolve_device


def build(cfg: DictConfig) -> tuple[LitSAE, ActivationStore]:
    device = resolve_device(cfg.device)
    tok = Tokenizer()
    model = load_pretrained(cfg.model_ckpt, device=device)  # arch read from the checkpoint
    model_cfg = model.cfg

    records = RecordDataset(
        open_or_build(cfg.data.fasta), tok, max_len=model_cfg.max_seq_len, fim_full_prob=cfg.data.fim_full_prob
    )
    record_loader = DataLoader(
        records, batch_size=cfg.data.record_batch_size, collate_fn=make_collate(tok.pad_id)
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
    # canonical SAE release (host_model + layer recorded): loads via idiom.IDiomSAE.from_pretrained
    save_sae(lit.sae, out_dir, host_model=str(cfg.model_ckpt), layer=cfg.layer)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sae")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
