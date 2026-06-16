"""Autoregressive training LightningModule — shared by pretraining and SFT.

The only difference between the two is the **loss mask** the data provides: pretraining
trains on every token (all-True mask); SFT trains only on the IDR completion
(``RecordDataset(completion_only=True)``). The module is otherwise identical, so SFT is just
"load a pretrained checkpoint + completion-only data + a gentler LR" (see ``configs/sft.yaml``).
"""

from __future__ import annotations

from dataclasses import asdict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm

from idiom.model.config import ModelConfig
from idiom.model.transformer import IDiomTransformer
from idiom.train.schedulers import warmup_cosine


class LitAutoregressive(L.LightningModule):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        lr: float = 3e-4,
        warmup_steps: int = 2000,
        max_steps: int = 100_000,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        min_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # Persist the architecture in the checkpoint so downstream loaders never need to
        # re-declare it (model/io.load_pretrained reads hparams["model_cfg"]).
        self.save_hyperparameters({"model_cfg": asdict(cfg)})
        self.model = IDiomTransformer(cfg)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps_ = max_steps
        self.weight_decay = weight_decay
        self.betas = betas
        self.min_lr_ratio = min_lr_ratio

    @classmethod
    def init_from_checkpoint(cls, ckpt_path: str, **kwargs) -> "LitAutoregressive":
        """Build a module and load model weights from a prior Lightning ckpt (for SFT).

        Architecture is read from the checkpoint (self-describing); never re-declared.
        """
        from idiom.model.io import config_from_checkpoint  # noqa: PLC0415

        lit = cls(config_from_checkpoint(ckpt_path), **kwargs)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        model_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
        lit.model.load_state_dict(model_state)
        return lit

    def _masked_loss(self, logits, targets, mask):
        # next-token CE, averaged only over the positions the mask selects (completion for SFT,
        # everything for pretraining); padded positions have mask=False so they never contribute.
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        return (per_token * mask).sum() / mask.sum().clamp(min=1)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        loss = self._masked_loss(self.model(x), y, mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        loss = self._masked_loss(self.model(x), y, mask)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Log per-parameter + total L2 gradient norms (grad_2.0_norm/*), matching the previous
        # IDiom version's pretrain logging.
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        sched = warmup_cosine(
            opt, warmup_steps=self.warmup_steps, max_steps=self.max_steps_, min_lr_ratio=self.min_lr_ratio
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
