"""Autoregressive training LightningModule shared by pretraining and SFT.

The only difference between the two is the loss mask the data provides: pretraining trains on
every token (all-True mask); SFT trains only on the IDR completion
(RecordDataset(completion_only=True)). The module is otherwise identical, so SFT is just loading
a pretrained checkpoint with completion-only data and a gentler learning rate (see
configs/sft.yaml).
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
    """Autoregressive next-token trainer for pretraining and SFT.

    Wraps an IDiomTransformer with a masked cross-entropy loss, AdamW, and a warmup-cosine
    learning-rate schedule. The data-provided loss mask selects which target positions
    contribute to the loss (all positions for pretraining, the IDR completion for SFT).

    Args:
        cfg (ModelConfig): Transformer architecture configuration.
        lr (float): Base learning rate.
        warmup_steps (int): Linear warmup length before cosine decay.
        max_steps (int): Total scheduler horizon (shared with the trainer).
        weight_decay (float): AdamW weight decay.
        betas (tuple[float, float]): AdamW beta coefficients.
        min_lr_ratio (float): Floor of the cosine decay as a fraction of the base LR.
    """

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
    def init_from_checkpoint(cls, init_from: str, **kwargs) -> "LitAutoregressive":
        """Build a module and warm-start its weights from a pretrained model (for SFT).

        init_from accepts any form model/io.load_model understands: a Lightning .ckpt, a released
        model directory, or a HuggingFace repo id (e.g. "jxliu2/idiom-20M"). The architecture is
        read from that artifact (self-describing), never re-declared.

        Args:
            init_from (str): A Lightning .ckpt, a released model directory, or a HF repo id.
            **kwargs: Optimizer and schedule hyperparameters forwarded to the constructor.

        Returns:
            LitAutoregressive: A module warm-started with the pretrained model's weights.
        """
        from idiom.model.io import load_model  # noqa: PLC0415

        model, cfg = load_model(init_from, eval_mode=False)
        lit = cls(cfg, **kwargs)
        lit.model.load_state_dict(model.state_dict())
        return lit

    def _masked_loss(self, logits, targets, mask):
        # next-token CE, averaged only over the positions the mask selects (completion for SFT,
        # everything for pretraining); padded positions have mask=False so they never contribute.
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        return (per_token * mask).sum() / mask.sum().clamp(min=1)

    def training_step(self, batch, batch_idx):
        """Compute and log the masked next-token loss for one training batch."""
        x, y, mask = batch
        loss = self._masked_loss(self.model(x), y, mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute and log the masked next-token loss for one validation batch."""
        x, y, mask = batch
        loss = self._masked_loss(self.model(x), y, mask)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Log per-parameter and total L2 gradient norms before each optimizer step."""
        # grad_2.0_norm/* keys, matching the earlier IDiom pretrain logging.
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self):
        """Build the AdamW optimizer and its per-step warmup-cosine schedule."""
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        sched = warmup_cosine(
            opt, warmup_steps=self.warmup_steps, max_steps=self.max_steps_, min_lr_ratio=self.min_lr_ratio
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
