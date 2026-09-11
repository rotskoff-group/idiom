"""Transformer configuration and presets for the three released model sizes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """IDiom transformer architecture settings.

    Attributes:
        vocab_size: Number of tokens in the tokenizer alphabet.
        n_layers: Number of transformer blocks.
        d_model: Residual-stream width; must be divisible by n_heads.
        n_heads: Number of attention heads.
        max_seq_len: Largest number of model positions, and the RoPE table length.
        rope_base: Base of the RoPE inverse-frequency progression.
        expansion_ratio: SwiGLU hidden width as a multiple of d_model.
        norm_eps: Epsilon used by every RMSNorm.
        qk_norm: If True, apply RMSNorm to queries and keys before rotation.
        tie_embeddings: If True, share the embedding matrix with the output head.

    Raises:
        ValueError: If d_model is not divisible by n_heads, or head_dim is odd.
    """

    vocab_size: int = 27 # idiom.data.tokenizer: 20 residues + 3 FIM markers + 4 controls
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    max_seq_len: int = 1024
    rope_base: float = 10_000.0
    expansion_ratio: float = 8 / 3
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
        """Per-head dimension, d_model // n_heads."""
        return self.d_model // self.n_heads


def idiom_20m() -> ModelConfig:
    """Architecture of jxliu2/idiom-20M (18.9M params)."""
    return ModelConfig(n_layers=6, d_model=512, n_heads=8)


def idiom_85m() -> ModelConfig:
    """Architecture of jxliu2/idiom-85M (85M params)."""
    return ModelConfig(n_layers=12, d_model=768, n_heads=12)


def idiom_300m() -> ModelConfig:
    """Architecture of jxliu2/idiom-300M (302M params)."""
    return ModelConfig(n_layers=24, d_model=1024, n_heads=16)
