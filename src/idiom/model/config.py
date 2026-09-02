"""Model hyperparameters for the IDiom transformer.

A flat dataclass; the named sizes at the bottom are just factory helpers for the three released
models, and every field is overridable. The architecture is the ESM-style block (RMSNorm, SwiGLU,
QK-norm, no bias, tied embeddings) with RoPE positions and no structural tokens.

A ModelConfig is never re-declared downstream: it travels inside every checkpoint and release
(see idiom.model.io), so loading an artifact recovers the architecture that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyperparameters defining an IDiom transformer's architecture."""

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


# The three released sizes, named for their parameter counts to match the Hub repo ids
# (jxliu2/idiom-20M, -85M, -300M). head_dim is 64 throughout.
def idiom_20m() -> ModelConfig:
    """Architecture of jxliu2/idiom-20M (18.9M params)."""
    return ModelConfig(n_layers=6, d_model=512, n_heads=8)


def idiom_85m() -> ModelConfig:
    """Architecture of jxliu2/idiom-85M (85M params)."""
    return ModelConfig(n_layers=12, d_model=768, n_heads=12)


def idiom_300m() -> ModelConfig:
    """Architecture of jxliu2/idiom-300M (302M params), the primary released model."""
    return ModelConfig(n_layers=24, d_model=1024, n_heads=16)
