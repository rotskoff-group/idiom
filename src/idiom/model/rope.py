"""Rotary position embeddings (RoPE, D7).

Llama-style "rotate-half" formulation. The cos/sin tables are precomputed up to
``max_seq_len`` and indexed by absolute ``positions`` — so a KV-cached decode step at
position ``p`` rotates with the same angles it would have during a full forward, which is
what makes cached and uncached generation agree.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rope(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        # inverse frequencies for each rotation plane (head_dim/2 of them)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, head_dim] (duplicated for rotate-half)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def apply(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate ``q``/``k`` (shape ``[B, H, L, head_dim]``) at the given absolute positions."""
        cos = self.cos[positions].to(q.dtype)[None, None]  # [1, 1, L, head_dim] -> broadcasts over B, H
        sin = self.sin[positions].to(q.dtype)[None, None]
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot
