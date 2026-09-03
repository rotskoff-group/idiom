"""Residual-stream activation extraction.

Runs the model with return_hidden_states, selects positions with the tokenizer's region mask, and
returns the kept activation rows together with the metadata needed to map each row back to the
sequence, position, and residue it came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from idiom.data.tokenizer import Tokenizer


@dataclass
class LayerActivations:
    """Kept activation rows at one layer, with per-row alignment metadata.

    Attributes:
        layer (int): The layer these activations were taken from.
        values (Tensor): Residual-stream vectors, shape [N_kept, d_model].
        seq_idx (Tensor): Row of the input batch each vector came from, shape [N_kept].
        pos_idx (Tensor): Position within that token sequence, shape [N_kept].
        token_id (Tensor): Token id at that position, shape [N_kept].
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
    """Extract residual-stream activations at the given layers for the selected positions.

    Positions are selected with Tokenizer.region_mask: control tokens are always dropped, and
    region restricts by position relative to the FIM "2" marker that opens the IDR.

    Args:
        model: The transformer to run.
        tokens (Tensor): Token ids of shape [B, L], as fed to the model.
        layers (list[int]): Layer indices whose residual stream to extract.
        tokenizer (Tokenizer | None): Tokenizer for position selection; a default if None.
        drop_markers (bool): If True, keep only real residues; if False, also keep FIM markers.
        region (str): Positions to keep: "all", "idr", or "non_idr".

    Returns:
        dict[int, LayerActivations]: One LayerActivations per requested layer, each holding the
            same selected positions in the same order.

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
