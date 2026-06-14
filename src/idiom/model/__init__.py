"""idiom.model — the IDiom transformer, RoPE, KV-cache attention, and sampling.

v2: sequence-only ``IDiomTransformer`` (structural tokens removed), RoPE-only
positional encoding, and KV-cached autoregressive decoding (prefill + single-token
steps) shared by inference and GRPO online generation.

Status: P2 — core landed (config, RoPE, RMSNorm, KV-cache attention, transformer). Sampling
+ the shared activation extractor (D14) are next. Migration source: legacy ``idiom.nn``.
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
