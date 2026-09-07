"""Residual-stream extraction with token and position metadata."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from idiom.data.tokenizer import Tokenizer


@dataclass
class LayerActivations:
    """Kept activation rows at one layer, with per-row alignment metadata.

    Attributes:
        layer: The layer these activations were taken from.
        values: Residual-stream vectors, shape [N_kept, d_model].
        seq_idx: Row of the input batch each vector came from, shape [N_kept].
        pos_idx: Position within that token sequence, shape [N_kept].
        token_id: Token id at that position, shape [N_kept].
    """

    layer: int
    values: Tensor
    seq_idx: Tensor
    pos_idx: Tensor
    token_id: Tensor


@torch.no_grad()
def extract_activations(
    model,
    tokens: Tensor,
    layers: list[int],
    *,
    tokenizer: Tokenizer | None = None,
    drop_markers: bool = True,
    region: str = "all",
) -> dict[int, LayerActivations]:
    """Extract selected residual-stream rows using Tokenizer.region_mask.

    Args:
        model: The transformer to run.
        tokens: Token ids of shape [B, L], as fed to the model.
        layers: Zero-based block indices.
        tokenizer: Tokenizer for position selection; a default if None.
        drop_markers: If True, keep only real residues; if False, also keep FIM markers.
        region: Positions to keep: "all", "idr", or "non_idr".

    Returns:
        One LayerActivations per requested layer, each holding the same selected positions in the
        same order.

    Raises:
        ValueError: If region is not "all", "idr", or "non_idr".
    """
    tok = tokenizer or Tokenizer()
    _, hidden = model(tokens, return_hidden_states=True)

    # Single shared selector (token class + IDR region) — identical to what steering uses.
    keep = tok.region_mask(tokens, region=region, drop_markers=drop_markers)  # [B, L] bool
    seq_idx, pos_idx = keep.nonzero(as_tuple=True)  # flat indices of kept tokens

    out: dict[int, LayerActivations] = {}
    for layer in layers:
        values = hidden[layer][seq_idx, pos_idx]  # [N_kept, d_model]
        out[layer] = LayerActivations(layer, values, seq_idx, pos_idx, tokens[seq_idx, pos_idx])
    return out
