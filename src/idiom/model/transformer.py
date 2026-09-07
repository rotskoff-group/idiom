"""Pre-norm RoPE transformer with optional tied embeddings and KV caching."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from idiom.model.attention import Attention, KVCache
from idiom.model.config import ModelConfig
from idiom.model.norms import RMSNorm
from idiom.model.rope import Rope


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network, with the gate and up projections fused into one matmul."""

    def __init__(self, d_model: int, expansion_ratio: float) -> None:
        """Build the fused gate/up projection and the down projection.

        Args:
            d_model: Input and output width.
            expansion_ratio: Hidden width as a multiple of d_model.
        """
        super().__init__()
        hidden = int(expansion_ratio * d_model)
        self.w_gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply down(silu(gate) * up), preserving shape [..., d_model]."""
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class Block(nn.Module):
    """Pre-norm transformer block computing x + attn(norm(x)) then x + ffn(norm(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the block's two norms, attention, and feed-forward network.

        Args:
            cfg: Architecture config for the attention and SwiGLU submodules.
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.expansion_ratio)

    def forward(self, x, rope, positions, cache=None, layer_idx=0):
        """Apply the attention and feed-forward sublayers with residual connections.

        Args:
            x (Tensor): Residual stream of shape [B, L, d_model].
            rope (Rope): Rotary embedding tables.
            positions (Tensor): Absolute position index per element of the L axis.
            cache (KVCache | None): Cache to read and extend, or None.
            layer_idx (int): This block's index, used as the cache slot.

        Returns:
            Tensor: The updated residual stream, shape [B, L, d_model].
        """
        x = x + self.attn(self.attn_norm(x), rope, positions, cache, layer_idx)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class IDiomTransformer(nn.Module):
    """Sequence-only pre-norm transformer with RoPE, tied embeddings, and KV-cache support.

    Attributes:
        cfg (ModelConfig): The architecture this model was built from.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the embedding, blocks, final norm, and output head, and initialize the weights.

        Linear and embedding weights are drawn from N(0, 0.02); the residual-stream output
        projections (attention wo and SwiGLU w_down) are rescaled by 1 / sqrt(2 * n_layers).

        Args:
            cfg: The architecture to build.
        """
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = Rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # GPT-style init: small std keeps init logits ~0, so initial CE ~ ln(vocab) instead of the
        # ~sqrt(d_model) blow-up from PyTorch's default Embedding std=1.0 (tied -> lm_head too).
        # Residual-stream writers (attn wo, ffn w_down) are scaled by 1/sqrt(2*n_layers) so the
        # residual variance doesn't grow with depth.
        for module in self.modules():  # not self.apply(): Rope defines its own .apply(q,k,positions)
            self._init_weights(module)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize one Linear or Embedding module in place."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor, *, cache: KVCache | None = None, return_hidden_states: bool = False):
        """Run the transformer, optionally through a KV cache and returning the residual stream.

        When a cache is given, positions continue from cache.length and the cache is extended by L.

        Args:
            tokens: Token ids of shape [B, L].
            cache: Cache to read and extend, or None for a full forward.
            return_hidden_states: If True, also return each block's residual-stream output.

        Returns:
            Tensor | tuple[Tensor, list[Tensor]]: Logits [B, L, vocab_size], or (logits, hidden)
                where hidden[i] is the residual stream after block i.
        """
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
