"""idiom.model — the IDiom transformer, RoPE, KV-cache attention, and sampling.

A sequence-only RMSNorm/SwiGLU/RoPE transformer with a KV cache (prefill plus single-token
decoding, shared by inference and GRPO online generation), plus sampling and the shared
residual-stream activation extractor.
"""

from idiom.model.attention import KVCache
from idiom.model.config import ModelConfig, idiom_12l, idiom_24l, idiom_36l
from idiom.model.sampling import generate, sample_next_token
from idiom.model.transformer import IDiomTransformer

__all__ = [
    "IDiomTransformer",
    "KVCache",
    "ModelConfig",
    "generate",
    "idiom_12l",
    "idiom_24l",
    "idiom_36l",
    "sample_next_token",
]
