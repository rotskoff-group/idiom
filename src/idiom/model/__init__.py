"""idiom.model — the IDiom transformer, RoPE, KV-cache attention, and sampling.

v2: sequence-only ``IDiomTransformer`` (structural tokens removed), RoPE-only
positional encoding, and KV-cached autoregressive decoding (prefill + single-token
steps) shared by inference and GRPO online generation.

Sequence-only RMSNorm/SwiGLU/RoPE transformer with a KV cache, plus sampling and the shared
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
