"""Multi-head self-attention with RoPE, QK-norm, and a KV cache.

Square attention is causal; cached single-token decoding attends to all keys.
No explicit padding mask is used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from idiom.model.config import ModelConfig
from idiom.model.norms import RMSNorm
from idiom.model.rope import Rope


class KVCache:
    """Per-layer key/value cache for autoregressive decoding.

    Attributes:
        k (list[Tensor | None]): Cached keys per layer, each [B, H, L, head_dim].
        v (list[Tensor | None]): Cached values per layer, same shape.
        length (int): Number of positions cached so far.
    """

    def __init__(self, n_layers: int) -> None:
        """Build an empty cache with one key/value slot per layer.

        Args:
            n_layers: Number of transformer layers to reserve slots for.
        """
        self.k: list[Tensor | None] = [None] * n_layers
        self.v: list[Tensor | None] = [None] * n_layers
        self.length = 0 # Advanced by the transformer once per forward

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append k and v ([B, H, L_new, head_dim]) to the selected layer in place.

        Return both tensors spanning past and new positions. The transformer advances length.
        """
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat((self.k[layer], k), dim=2)
            self.v[layer] = torch.cat((self.v[layer], v), dim=2)
        return self.k[layer], self.v[layer]


class Attention(nn.Module):
    """Multi-head self-attention with RoPE, QK-norm, and KV caching."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Initialize attention projections and optional QK norms.

        Args:
            cfg: Architecture config supplying n_heads, head_dim, d_model, qk_norm, and norm_eps.
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        rope: Rope,
        positions: Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> Tensor:
        """Attend over x, rotating q/k to the given absolute positions.

        Args:
            x: Normalized input of shape [B, L, d_model].
            rope: Rotary embedding tables.
            positions: Absolute position index per element of the L axis.
            cache: Cache to read and extend, or None for a full forward.
            layer_idx: This layer's index, used as the cache slot.

        Returns:
            The attention output of shape [B, L, d_model].
        """
        B, L, _ = x.shape
        qkv = self.wqkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2)) # each [B, H, L, head_dim]

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope.apply(q, k, positions)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(q.size(2) == k.size(2)))
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)
