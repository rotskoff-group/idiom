"""LightningModule wrapping a SparseCoder.

The training recipe is faithful to EleutherAI sparsify (and OpenAI sparse_autoencoder):

- loss = fvu + auxk_alpha * auxk_loss + multi_topk_fvu / 8 (all computed in the SAE);
- decoder rows renormalized to unit norm before every forward;
- decoder gradient component parallel to its rows projected out before the optimizer step;
- dead latents = not fired in the last dead_feature_tokens tokens; AuxK only when any;
- Adam with LR auto-scaled 2e-4 / (num_latents / 2**14)**0.5 (sparsify) and linear warmup;
- no gradient clipping by default (matches sparsify's fvu path).
"""

from __future__ import annotations

import lightning as L
import torch as t

from idiom.sae.sparse_coder import SparseCoder


def _lr_lambda(total_steps: int, warmup_steps: int, decay_start: int | None):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        if decay_start is not None and step >= decay_start and decay_start < total_steps:
            return (total_steps - step) / (total_steps - decay_start)
        return 1.0

    return fn


class LitSAE(L.LightningModule):
    """LightningModule that trains a SparseCoder with the sparsify FVU / AuxK recipe."""

    def __init__(
        self,
        d_in: int,
        *,
        k: int = 32,
        expansion_factor: int = 8,
        activation: str = "topk",
        multi_topk: bool = False,
        normalize_decoder: bool = True,
        lr: float | None = None,
        total_steps: int = 100_000,
        warmup_steps: int = 1_000,
        decay_start: int | None = None,
        auxk_alpha: float = 1 / 32,
        dead_feature_tokens: int = 10_000_000,
        grad_clip_norm: float | None = None,
    ):
        """Build the LightningModule and its SparseCoder.

        Args:
            d_in (int): Input (residual-stream) dimension.
            k (int): Number of latents kept active per token.
            expansion_factor (int): Latents-per-input multiplier.
            activation (str): Selection rule, "topk" or "groupmax".
            multi_topk (bool): If True, add the Multi-TopK auxiliary loss.
            normalize_decoder (bool): If True, keep decoder rows at unit norm.
            lr (float | None): Learning rate; if None, auto-scaled from num_latents (sparsify).
            total_steps (int): Total optimizer steps, used by the LR schedule.
            warmup_steps (int): Linear warmup steps at the start of training.
            decay_start (int | None): Step at which linear LR decay begins (no decay if None).
            auxk_alpha (float): Weight on the AuxK dead-latent revival loss.
            dead_feature_tokens (int): A latent is dead if it has not fired in this many tokens.
            grad_clip_norm (float | None): Gradient-norm clip value; no clipping if None.
        """
        super().__init__()
        self.save_hyperparameters()

        self.sae = SparseCoder(
            d_in,
            expansion_factor=expansion_factor,
            k=k,
            activation=activation,
            multi_topk=multi_topk,
            normalize_decoder=normalize_decoder,
        )

        # LR auto-scaling from sparsify (smaller LR for wider dictionaries).
        if lr is None:
            lr = 2e-4 / (self.sae.num_latents / (2**14)) ** 0.5
        self.lr = lr

        self.auxk_alpha = auxk_alpha
        self.dead_feature_tokens = dead_feature_tokens
        self.grad_clip_norm = grad_clip_norm

        self.register_buffer(
            "num_tokens_since_fired",
            t.zeros(self.sae.num_latents, dtype=t.long),
            persistent=False,
        )

    @t.no_grad()
    def init_b_dec_from_mean(self, mean_activation: t.Tensor):
        """Initialize the decoder bias to the data mean (sparsify init).

        Args:
            mean_activation (t.Tensor): The mean activation vector of shape [d_in].
        """
        self.sae.b_dec.data = mean_activation.to(self.sae.b_dec.device, self.sae.b_dec.dtype)

    def on_train_batch_start(self, *args, **kwargs):
        """Re-impose the decoder unit-norm constraint before each forward (sparsify ordering).

        The optimizer step can push rows off the unit sphere, and a latent's activation is only
        interpretable as a magnitude along a *unit* direction, so the constraint is restored before
        the rows are used rather than after they are updated.
        """
        if self.sae.normalize_decoder:
            self.sae.set_decoder_norm_to_unit_norm()

    def training_step(self, batch: t.Tensor, batch_idx: int):
        """Run one SAE step on a batch of activations and update the dead-latent counters.

        Args:
            batch (t.Tensor): Activation rows of shape [n_tokens, d_in].
            batch_idx (int): Index of the batch within the epoch (unused).

        Returns:
            t.Tensor: fvu + auxk_alpha * auxk_loss + multi_topk_fvu / 8.
        """
        dead_mask = (
            self.num_tokens_since_fired > self.dead_feature_tokens if self.auxk_alpha > 0 else None
        )
        out = self.sae(batch, dead_mask=dead_mask)
        loss = out.fvu + self.auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8

        # dead-feature bookkeeping (in tokens)
        n_tok = batch.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[out.latent_indices.flatten()] = True
        self.num_tokens_since_fired += n_tok
        self.num_tokens_since_fired[did_fire] = 0
        n_dead = int((self.num_tokens_since_fired > self.dead_feature_tokens).sum())

        l0 = (out.latent_acts > 0).float().sum(-1).mean()
        self.log_dict(
            {
                "train/loss": loss,
                "train/fvu": out.fvu,
                "train/explained_variance": 1.0 - out.fvu,
                "train/auxk_loss": out.auxk_loss,
                "train/multi_topk_fvu": out.multi_topk_fvu,
                "train/l0": l0,
                "train/dead_features": float(n_dead),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=n_tok,
        )
        return loss

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """Project the decoder gradient onto the unit sphere's tangent space before the step.

        A gradient component parallel to a unit-norm decoder row only changes that row's length,
        which the renorm then undoes — so it contributes nothing but does perturb the optimizer's
        momentum estimates. Removing it here (sparsify's ordering: after backward, before the step)
        keeps Adam's statistics about the directions that actually move. Lightning routes this
        through the clipping hook because that is the one callback between the two.
        """
        if self.sae.normalize_decoder and self.sae.W_dec.grad is not None:
            self.sae.remove_gradient_parallel_to_decoder_directions()
        if self.grad_clip_norm is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.grad_clip_norm,
                gradient_clip_algorithm="norm",
            )

    @t.no_grad()
    def validation_step(self, batch: t.Tensor, batch_idx: int):
        """Log held-out FVU / explained variance / L0 (no AuxK: dead-latent revival is training-only).

        Args:
            batch (t.Tensor): Activation rows of shape [n_tokens, d_in].
            batch_idx (int): Index of the batch within the epoch (unused).
        """
        out = self.sae(batch)
        l0 = (out.latent_acts > 0).float().sum(-1).mean()
        self.log_dict(
            {
                "val/fvu": out.fvu,
                "val/explained_variance": 1.0 - out.fvu,
                "val/l0": l0,
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.size(0),
        )

    def configure_optimizers(self):
        """Build Adam plus the per-step warmup / linear-decay schedule."""
        opt = t.optim.Adam(self.sae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        sched = t.optim.lr_scheduler.LambdaLR(
            opt,
            _lr_lambda(
                self.hparams.total_steps, self.hparams.warmup_steps, self.hparams.decay_start
            ),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }
