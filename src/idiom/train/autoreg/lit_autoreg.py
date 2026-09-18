"""Lightning training with masked next-token cross-entropy."""

from __future__ import annotations

from dataclasses import asdict

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from idiom.model.config import ModelConfig
from idiom.model.io import load_model
from idiom.model.transformer import IDiomTransformer
from idiom.train.autoreg.schedulers import warmup_cosine


class LitAutoregressive(L.LightningModule):
    """Train an IDiom transformer with masked next-token cross-entropy.

    Uses AdamW with a warmup-cosine learning-rate schedule, and stores the ModelConfig in its
    hyperparameters.

    Attributes:
        cfg (ModelConfig): The architecture being trained.
        model (IDiomTransformer): The wrapped transformer.
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
        """Initialize the transformer and training settings.

        Args:
            cfg: Transformer architecture configuration.
            lr: Base learning rate.
            warmup_steps: Linear warmup length before the cosine decay begins.
            max_steps: Scheduler horizon, normally the trainer's max_steps.
            weight_decay: AdamW weight decay.
            betas: AdamW beta coefficients.
            min_lr_ratio: Floor of the cosine decay, as a fraction of the base learning rate.
        """
        super().__init__()
        self.cfg = cfg
        # Store the architecture so checkpoint loaders can reconstruct the model
        self.save_hyperparameters({"model_cfg": asdict(cfg)})
        self.model = IDiomTransformer(cfg)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps_ = max_steps
        self.weight_decay = weight_decay
        self.betas = betas
        self.min_lr_ratio = min_lr_ratio

    @classmethod
    def init_from_checkpoint(cls, init_from: str, **kwargs) -> LitAutoregressive:
        """Initialize training from pretrained weights.

        The architecture is read from the artifact.

        Args:
            init_from: A Lightning .ckpt, a released model directory, or a Hub repo id; any form
                idiom.model.io.load_model accepts.
            **kwargs: Optimizer and schedule arguments forwarded to the constructor.

        Returns:
            A module holding the pretrained weights.
        """
        model, cfg = load_model(init_from, eval_mode=False)
        lit = cls(cfg, **kwargs)
        lit.model.load_state_dict(model.state_dict())
        return lit

    def _masked_loss(self, logits, targets, mask) -> torch.Tensor:
        """Return the cross-entropy averaged over the positions the mask selects."""
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        return (per_token * mask).sum() / mask.sum().clamp(min=1)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Return and log masked cross-entropy for (input_ids, target_ids, loss_mask)."""
        x, y, mask = batch
        loss = self._masked_loss(self.model(x), y, mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Return and log validation cross-entropy for (input_ids, target_ids, loss_mask)."""
        x, y, mask = batch
        loss = self._masked_loss(self.model(x), y, mask)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log per-parameter and total L2 gradient norms."""
        self.log_dict(grad_norm(self, norm_type=2))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return AdamW with a per-step warmup-cosine schedule."""
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        sched = warmup_cosine(
            opt, warmup_steps=self.warmup_steps, max_steps=self.max_steps_, min_lr_ratio=self.min_lr_ratio
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
