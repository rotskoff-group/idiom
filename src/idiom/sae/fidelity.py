"""Downstream fidelity of an SAE: how much of IDiom's next-token prediction survives when a
layer's residual stream is replaced by the SAE reconstruction.

The Gao-style "loss recovered" metric, on IDiom's autoregressive objective. For each batch we
compute next-token NLL three ways at the SAE's layer (via forward hooks on the block):

    loss_clean   = NLL with the original residual stream
    loss_sae     = NLL with residual replaced by sae(residual)
    loss_ablate  = NLL with residual replaced by a baseline (mean / b_dec)
    pct_recovered = (loss_ablate - loss_sae) / (loss_ablate - loss_clean) * 100

A perfect SAE recovers ~100%; the information-free baseline recovers 0%.

By default (``residue_only=True``) the substitution and ablation baseline are applied only at
real-residue positions — the ones the SAE was trained on. The extractor drops START / FIM-marker
/ control activations, so editing them feeds the SAE out-of-distribution inputs that, through
causal attention, corrupt predictions across the sequence and badly understate fidelity (e.g.
21% vs 80% recovered for the same checkpoint). See :class:`idiom.data.tokenizer.Tokenizer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.sae.steering.hooks import sae_edit_hook, steering, substitute_hook


@dataclass
class FidelityResult:
    loss_clean: float
    loss_sae: float
    loss_ablate: float

    @property
    def pct_loss_recovered(self) -> float:
        denom = self.loss_ablate - self.loss_clean
        if abs(denom) < 1e-8:
            return float("nan")
        return (self.loss_ablate - self.loss_sae) / denom * 100.0


def _identity_edit(f: torch.Tensor) -> torch.Tensor:
    return f


@torch.no_grad()
def compute_fidelity(
    model,
    sae,
    layer: int,
    batches,
    *,
    pad_id: int,
    tokenizer: Tokenizer | None = None,
    baseline: torch.Tensor | None = None,
    residue_only: bool = True,
    device: str | torch.device | None = None,
) -> FidelityResult:
    """Substitution-loss fidelity over ``(input, target, mask)`` batches (``RecordDataset``).

    Args:
        model: an :class:`IDiomTransformer` (``model(tokens) -> logits``, ``.blocks`` hook points).
        sae: trained SAE for ``layer`` (uses ``encode_dense`` / ``decode_dense`` / ``b_dec``).
        layer: residual-stream layer the SAE was trained on.
        batches: iterable of ``(input, target, loss_mask)``.
        pad_id: ignore index for the next-token loss (tokenizer PAD = 23).
        tokenizer: required when ``residue_only`` (builds the residue mask).
        baseline: ablation vector; defaults to ``sae.b_dec`` (mean-ablation).
        residue_only: confine the SAE/ablation edit to real-residue positions (default True).
    """
    device = device or next(model.parameters()).device
    sae = sae.to(device).eval()
    tok = tokenizer or Tokenizer()
    if baseline is None:
        baseline = sae.b_dec.detach()
    baseline = baseline.to(device)

    ce = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    sums = {"clean": 0.0, "sae": 0.0, "ablate": 0.0}
    n_tokens = 0
    mask_tok = tok if residue_only else None  # passed to steering -> per-forward residue mask

    def nll(x: torch.Tensor, y: torch.Tensor) -> float:
        return ce(model(x).permute(0, 2, 1), y).item()

    for x, y, _ in batches:
        x, y = x.to(device), y.to(device)
        n_tokens += int((y != pad_id).sum().item())

        sums["clean"] += nll(x, y)
        with steering(model, layer, sae_edit_hook(sae, _identity_edit), tokenizer=mask_tok):
            sums["sae"] += nll(x, y)
        with steering(model, layer, substitute_hook(baseline), tokenizer=mask_tok):
            sums["ablate"] += nll(x, y)

    if n_tokens == 0:
        raise ValueError("No non-pad target tokens found in the provided batches.")

    return FidelityResult(
        loss_clean=sums["clean"] / n_tokens,
        loss_sae=sums["sae"] / n_tokens,
        loss_ablate=sums["ablate"] / n_tokens,
    )
