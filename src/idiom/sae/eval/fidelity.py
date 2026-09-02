"""Substitution-loss fidelity of an SAE.

For each batch the next-token NLL is computed three ways, using forward hooks on the SAE's layer:

    loss_clean   = NLL with the original residual stream
    loss_sae     = NLL with the residual replaced by sae(residual)
    loss_ablate  = NLL with the residual replaced by a fixed baseline vector
    pct_recovered = (loss_ablate - loss_sae) / (loss_ablate - loss_clean) * 100

A perfect reconstruction recovers 100 percent, and the information-free baseline recovers 0. Both
the substitution and the ablation are confined to the SAE's training region.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.sae.steering.hooks import sae_edit_hook, steering, substitute_hook


@dataclass
class FidelityResult:
    """The three mean next-token losses of a fidelity run.

    Attributes:
        loss_clean (float): Mean NLL with the original residual stream.
        loss_sae (float): Mean NLL with the residual replaced by its SAE reconstruction.
        loss_ablate (float): Mean NLL with the residual replaced by the baseline vector.
    """

    loss_clean: float
    loss_sae: float
    loss_ablate: float

    @property
    def pct_loss_recovered(self) -> float:
        """Percent of the ablated loss recovered by the SAE.

        Returns:
            float: (loss_ablate - loss_sae) / (loss_ablate - loss_clean) * 100, or NaN when the
                clean and ablated losses are equal.
        """
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
    """Compute substitution-loss fidelity over batches of (input, target, mask) triples.

    Each batch is run three times, and the losses are summed over batches and divided by the total
    number of non-pad target tokens.

    Args:
        model: An IDiomTransformer, called as model(tokens) -> logits, with .blocks as hook points.
        sae: The trained SAE for layer, providing encode_dense, decode_dense, and b_dec.
        layer (int): The residual-stream layer the SAE was trained on.
        batches: Iterable of (input, target, loss_mask) triples; the mask is unused.
        pad_id (int): Ignore index for the next-token loss.
        tokenizer (Tokenizer | None): Builds the per-forward region mask; a default if None.
        baseline (torch.Tensor | None): The ablation vector; sae.b_dec if None.
        region (str): Positions to edit: "all", "idr", or "non_idr".
        device (str | torch.device | None): Device to run on; the model's device if None.

    Returns:
        FidelityResult: The mean clean, SAE-substituted, and ablated next-token losses.

    Raises:
        ValueError: If the batches contain no non-pad target tokens.
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
