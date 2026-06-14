"""Model hyperparameters for the IDiom transformer (v2).

Flat dataclass (D6). Named sizes (12L test/de-risk → 24L primary → 36L) are just factory
helpers; everything is overridable. The architecture is the proven ESM-style block —
**RMSNorm + SwiGLU + QK-norm, no bias, tied embeddings** — with **RoPE** positions and no
structural tokens (D7, D8, D17).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 27  # idiom.data.tokenizer: 20 residues + 3 FIM markers + 4 controls
    n_layers: int = 12  # defaults = GPT-2 small (d768/12h, head_dim 64)
    d_model: int = 768
    n_heads: int = 12
    max_seq_len: int = 1024
    rope_base: float = 10_000.0
    expansion_ratio: float = 8 / 3  # SwiGLU hidden ≈ expansion_ratio * d_model
    norm_eps: float = 1e-5
    qk_norm: bool = True
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# Named sizes = the GPT-2 family (head_dim 64 throughout); train 12L first, then scale.
def idiom_12l() -> ModelConfig:  # GPT-2 small
    return ModelConfig(n_layers=12, d_model=768, n_heads=12)


def idiom_24l() -> ModelConfig:  # GPT-2 medium, the primary retrain
    return ModelConfig(n_layers=24, d_model=1024, n_heads=16)


def idiom_36l() -> ModelConfig:  # GPT-2 large
    return ModelConfig(n_layers=36, d_model=1280, n_heads=20)
