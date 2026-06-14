"""The IDiom transformer: sequence-only, pre-norm, RoPE, KV-cache-ready (D8, D17).

``forward`` serves both training (full sequence, no cache) and autoregressive decoding (pass a
:class:`KVCache`; positions continue from ``cache.length``). With ``return_hidden_states`` it
also returns each block's residual-stream output — the activations the SAE / extractor consume.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from idiom.model.attention import Attention, KVCache
from idiom.model.config import ModelConfig
from idiom.model.norms import RMSNorm
from idiom.model.rope import Rope


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate) * up)``, gate+up fused in one projection."""

    def __init__(self, d_model: int, expansion_ratio: float) -> None:
        super().__init__()
        hidden = int(expansion_ratio * d_model)
        self.w_gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.expansion_ratio)

    def forward(self, x, rope, positions, cache=None, layer_idx=0):
        x = x + self.attn(self.attn_norm(x), rope, positions, cache, layer_idx)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class IDiomTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, tokens: Tensor, *, cache: KVCache | None = None, return_hidden_states: bool = False):
        B, L = tokens.shape
        past = cache.length if cache is not None else 0
        positions = torch.arange(past, past + L, device=tokens.device)

        x = self.embed(tokens)
        hidden: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, positions, cache, i)
            if return_hidden_states:
                hidden.append(x)  # residual stream after block i (what the SAE trains on)

        if cache is not None:
            cache.length += L  # advance once per forward, after every layer has appended

        logits = self.lm_head(self.final_norm(x))
        return (logits, hidden) if return_hidden_states else logits
