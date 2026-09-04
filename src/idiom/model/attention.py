"""Multi-head self-attention with RoPE, QK-norm, and a KV cache.

A square attention block (q_len == kv_len, i.e. training or prefill) is masked causally; any other
shape, such as a cached decode step, attends to everything. No explicit padding mask is used.
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
            n_layers (int): Number of transformer layers to reserve slots for.
        """
        self.k: list[Tensor | None] = [None] * n_layers
        self.v: list[Tensor | None] = [None] * n_layers
        self.length = 0  # tokens cached so far (advanced by the transformer, once per forward)

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append this step's keys/values for one layer and return the full cached history.

        Args:
            layer (int): Layer index whose cache to extend.
            k (Tensor): New keys of shape [B, H, L_new, head_dim].
            v (Tensor): New values of the same shape.

        Returns:
            tuple[Tensor, Tensor]: The keys and values spanning past + current positions.
        """
        # Append the new keys/values along the sequence dim (=2 for [B, H, L, head_dim]).
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat((self.k[layer], k), dim=2)
            self.v[layer] = torch.cat((self.v[layer], v), dim=2)
        return self.k[layer], self.v[layer]


class Attention(nn.Module):
    """Multi-head self-attention with RoPE, QK-norm, and an optional KV cache."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the fused QKV and output projections, and the optional QK norms.

        Args:
            cfg (ModelConfig): Architecture config supplying n_heads, head_dim, d_model, qk_norm,
                and norm_eps.
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
            x (Tensor): Normalized input of shape [B, L, d_model].
            rope (Rope): Rotary embedding tables.
            positions (Tensor): Absolute position index per element of the L axis.
            cache (KVCache | None): Cache to read and extend, or None for a full forward.
            layer_idx (int): This layer's index, used as the cache slot.

        Returns:
            Tensor: The attention output of shape [B, L, d_model].
        """
        B, L, _ = x.shape
        qkv = self.wqkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))  # each [B, H, L, head_dim]

        q, k = self.q_norm(q), self.k_norm(k)  # QK-norm before rotation
        q, k = rope.apply(q, k, positions)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)  # k/v now span past + current

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(q.size(2) == k.size(2)))
        out = out.transpose(1, 2).reshape(B, L, -1)  # [B, L, d_model]
        return self.wo(out)
