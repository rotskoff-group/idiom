"""Rotary position embeddings (RoPE), in the Llama-style rotate-half formulation.

The cos/sin tables are precomputed up to max_seq_len and indexed by absolute position.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope(nn.Module):
    """Rotary position embeddings with precomputed cos/sin tables."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        """Precompute the cos and sin tables as non-persistent buffers.

        Args:
            head_dim (int): Per-head dimension; must be even.
            max_seq_len (int): Largest position the tables cover.
            base (float): Base of the inverse-frequency geometric progression.
        """
        super().__init__()
        # inverse frequencies for each rotation plane (head_dim/2 of them)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, head_dim] (duplicated for rotate-half)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def apply(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate q and k at the given absolute positions.

        Args:
            q (Tensor): Query tensor of shape [B, H, L, head_dim].
            k (Tensor): Key tensor of shape [B, H, L, head_dim].
            positions (Tensor): Absolute position index for each element of the L axis.

        Returns:
            tuple[Tensor, Tensor]: The rotated q and k, with the same shapes as the inputs.
        """
        cos = self.cos[positions].to(q.dtype)[None, None]  # [1, 1, L, head_dim] -> broadcasts over B, H
        sin = self.sin[positions].to(q.dtype)[None, None]
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot
