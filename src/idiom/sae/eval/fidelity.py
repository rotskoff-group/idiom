"""Downstream fidelity of an SAE: how much of IDiom's next-token prediction survives when a
layer's residual stream is replaced by the SAE reconstruction.

The Gao-style "loss recovered" metric, on IDiom's autoregressive objective. For each batch we
compute next-token NLL three ways at the SAE's layer (via forward hooks on the block):

    loss_clean   = NLL with the original residual stream
    loss_sae     = NLL with residual replaced by sae(residual)
    loss_ablate  = NLL with residual replaced by a baseline (mean / b_dec)
    pct_recovered = (loss_ablate - loss_sae) / (loss_ablate - loss_clean) * 100

A perfect SAE recovers ~100%; the information-free baseline recovers 0%.

The substitution and ablation baseline are applied only at the SAE's training region (via the
tokenizer's region_mask) — the positions the SAE was trained on. The extractor drops START /
FIM-marker / control activations, so editing them feeds the SAE out-of-distribution inputs that,
through causal attention, corrupt predictions across the sequence and badly understate fidelity
(e.g. 21% vs 80% recovered for the same checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.sae.steering.hooks import sae_edit_hook, steering, substitute_hook


@dataclass
class FidelityResult:
    """The three mean next-token losses of a fidelity run and the percent-recovered they imply."""

    loss_clean: float
    loss_sae: float
    loss_ablate: float

    @property
    def pct_loss_recovered(self) -> float:
        """Return the percent of loss recovered: (ablate - sae) / (ablate - clean) * 100."""
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
    region: str = "all",
    device: str | torch.device | None = None,
) -> FidelityResult:
    """Compute substitution-loss fidelity over (input, target, mask) batches from a RecordDataset.

    Args:
        model: An IDiomTransformer (model(tokens) -> logits, with .blocks hook points).
        sae: The trained SAE for layer (uses encode_dense / decode_dense / b_dec).
        layer (int): The residual-stream layer the SAE was trained on.
        batches: Iterable of (input, target, loss_mask) triples.
        pad_id (int): Ignore index for the next-token loss (tokenizer PAD = 23).
        tokenizer (Tokenizer | None): Builds the per-forward region mask (fresh Tokenizer if None).
        baseline (torch.Tensor | None): Ablation vector; defaults to sae.b_dec (mean-ablation).
        region (str): The SAE's training region ("all", "idr", or "non_idr"). The SAE / ablation
            edit is always confined to it — editing positions the SAE never saw (markers, START, the
            wrong side of the "2") feeds it out-of-distribution inputs that corrupt predictions
            through causal attention and badly understate fidelity.
        device (str | torch.device | None): Device to run on (model's device if None).

    Returns:
        FidelityResult: The mean clean, SAE-substituted, and ablated next-token losses.

    Raises:
        ValueError: If no non-pad target tokens are found in the provided batches.
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

    def nll(x: torch.Tensor, y: torch.Tensor) -> float:
        return ce(model(x).permute(0, 2, 1), y).item()

    for x, y, _ in batches:
        x, y = x.to(device), y.to(device)
        n_tokens += int((y != pad_id).sum().item())

        sums["clean"] += nll(x, y)
        with steering(model, layer, sae_edit_hook(sae, _identity_edit), tokenizer=tok, region=region):
            sums["sae"] += nll(x, y)
        with steering(model, layer, substitute_hook(baseline), tokenizer=tok, region=region):
            sums["ablate"] += nll(x, y)

    if n_tokens == 0:
        raise ValueError("No non-pad target tokens found in the provided batches.")

    return FidelityResult(
        loss_clean=sums["clean"] / n_tokens,
        loss_sae=sums["sae"] / n_tokens,
        loss_ablate=sums["ablate"] / n_tokens,
    )
