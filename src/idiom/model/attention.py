"""Multi-head self-attention with RoPE, QK-norm, and a KV cache (D17).

Causal masking: training/prefill process a square ``[L, L]`` block, so we let SDPA apply the
causal mask (``is_causal=True``). With right-padding + causal, real tokens never attend to
pad positions (pad is always to their right), so no explicit pad mask is needed. A KV-cached
decode step has one query against ``N`` cached keys (``is_causal=False`` — it should see all
past). The rule below is just ``q_len == kv_len`` → causal, else attend-all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from idiom.model.config import ModelConfig
from idiom.model.norms import RMSNorm
from idiom.model.rope import Rope


class KVCache:
    """Per-layer key/value cache for autoregressive decoding."""

    def __init__(self, n_layers: int) -> None:
        self.k: list[Tensor | None] = [None] * n_layers
        self.v: list[Tensor | None] = [None] * n_layers
        self.length = 0  # tokens cached so far (advanced by the transformer, once per forward)

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # Append the new keys/values along the sequence dim (=2 for [B, H, L, head_dim]).
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat((self.k[layer], k), dim=2)
            self.v[layer] = torch.cat((self.v[layer], v), dim=2)
        return self.k[layer], self.v[layer]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
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
