"""Pure-Python helpers for extract_activations — FIM filter + segment walk.

Kept in a separate module so they can be imported (and unit-tested) without pulling
in the heavy ``LightningModel`` / ``pl_bolts`` import chain that ``extract_activations``
needs for the actual extraction.
"""

from __future__ import annotations

import numpy as np
import torch


def _alphabet_chars(token_info) -> list[str]:
    """Decode the saved alphabet to a list of single-character strings indexed by token id."""
    alphabet = token_info["alphabet"]
    return [
        a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
        for a in alphabet
    ]


def _get_fim_token_ids(token_info) -> dict[int, int]:
    """Map FIM marker token ids ('1', '3', '2') to their segment labels (1, 3, 2).

    FIM format: ``1{prefix}3{suffix}2{IDR}``. Residues between '1' and '3' belong to
    segment 1 (prefix); between '3' and '2' to segment 3 (suffix); after '2' to
    segment 2 (IDR). The label equals the literal marker character for readability.
    """
    chars = _alphabet_chars(token_info)
    out = {}
    for ch, label in (("1", 1), ("3", 3), ("2", 2)):
        if ch not in chars:
            raise ValueError(f"FIM marker {ch!r} not found in alphabet {chars!r}")
        out[chars.index(ch)] = label
    return out


def _get_ctrl_token_ids(token_info) -> torch.Tensor:
    """Token ids for PAD/START/STOP/MASK — filtered from saved activations/tokens."""
    return torch.tensor(
        [int(v) for k, v in token_info["input"]["TOK"].items() if k != "TOK_MAX_SIZE"],
        dtype=torch.long,
    )


def _get_filter_token_ids(token_info) -> tuple[torch.Tensor, dict[int, int]]:
    """Token ids excluded from saved activations: PAD/START/STOP/MASK + FIM 1/3/2.

    Returns ``(filter_ids, fim_id_to_label)``.
    """
    ctrl_ids = [
        int(v) for k, v in token_info["input"]["TOK"].items() if k != "TOK_MAX_SIZE"
    ]
    fim_id_to_label = _get_fim_token_ids(token_info)
    filter_ids = torch.tensor(ctrl_ids + list(fim_id_to_label), dtype=torch.long)
    return filter_ids, fim_id_to_label


def _compute_fim_segments(
    src_tokens: torch.Tensor,
    fim_id_to_label: dict[int, int],
    ctrl_token_ids: torch.Tensor,
    global_seq_offset: int,
) -> np.ndarray:
    """Per-position FIM segment labels for a batch of ``src_tokens`` ([B, L] int).

    Walks each sequence left-to-right tracking the most recently seen FIM marker.
    Residues get the label of that marker. Marker positions themselves are labeled
    too (they get filtered out of saved activations anyway). Ctrl-token positions
    get 0.

    Raises ``ValueError`` if a residue is encountered before any FIM marker — every
    kept residue must follow a marker.
    """
    src_np = src_tokens.numpy() if isinstance(src_tokens, torch.Tensor) else src_tokens
    B, L = src_np.shape
    fim_ids = set(fim_id_to_label)
    ctrl_set = set(int(x) for x in ctrl_token_ids.tolist())
    SENTINEL = -1
    out = np.zeros((B, L), dtype=np.int8)
    for b in range(B):
        current = SENTINEL
        for i in range(L):
            tid = int(src_np[b, i])
            if tid in fim_ids:
                current = fim_id_to_label[tid]
                out[b, i] = current
            elif tid in ctrl_set:
                out[b, i] = 0
            else:
                if current == SENTINEL:
                    raise ValueError(
                        f"Residue token id {tid} at sequence "
                        f"{global_seq_offset + b}, position {i} appears before any "
                        f"FIM marker. Every kept residue must follow a FIM marker."
                    )
                out[b, i] = current
    return out
