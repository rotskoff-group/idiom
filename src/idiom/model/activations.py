"""Residual-stream activation extraction — the one core behind SAE training and user export (D14).

Runs the model with ``return_hidden_states`` and selects the residue positions (dropping FIM
markers + control tokens via the tokenizer), returning the kept activation rows together with
the alignment metadata (which sequence, which position, which residue) needed to map every
vector back to its residue. The SAE stream (P5) and the ``extract`` export tool (P5) both call
this, so what the SAE trains on and what users save are identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from idiom.data.tokenizer import Tokenizer


@dataclass
class LayerActivations:
    layer: int
    values: Tensor  # [N_kept, d_model] residual-stream vectors
    seq_idx: Tensor  # [N_kept] row in the input batch each vector came from
    pos_idx: Tensor  # [N_kept] position within that token sequence
    token_id: Tensor  # [N_kept] residue token id (residue identity; map via tokenizer.decode)


@torch.no_grad()
def extract_activations(
    model,
    tokens: Tensor,
    layers: list[int],
    *,
    tokenizer: Tokenizer | None = None,
    drop_markers: bool = True,
) -> dict[int, LayerActivations]:
    """Extract residual-stream activations at ``layers`` for the kept positions.

    ``tokens`` is ``[B, L]`` token ids (as fed to the model, i.e. START-prefixed, right-padded).
    ``drop_markers=True`` keeps only real residues (the SAE default); ``False`` keeps residues +
    FIM markers. Control tokens (START/STOP/PAD/MASK) are always dropped.
    """
    tok = tokenizer or Tokenizer()
    _, hidden = model(tokens, return_hidden_states=True)

    # Residues are ids < n_residues; residues+markers are ids < n_residues+n_fim; controls are
    # the rest (>= 23) and are always excluded.
    cutoff = tok.n_residues if drop_markers else tok.n_residues + tok.n_fim
    keep = tokens < cutoff  # [B, L] bool
    seq_idx, pos_idx = keep.nonzero(as_tuple=True)  # flat indices of kept tokens

    out: dict[int, LayerActivations] = {}
    for layer in layers:
        values = hidden[layer][seq_idx, pos_idx]  # [N_kept, d_model]
        out[layer] = LayerActivations(layer, values, seq_idx, pos_idx, tokens[seq_idx, pos_idx])
    return out
