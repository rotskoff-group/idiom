"""Model layer: the IDiom transformer, its components, sampling, and activation extraction.

Modules:
    config: the ModelConfig architecture dataclass and the released model sizes.
    norms: RMSNorm.
    rope: rotary position embeddings.
    attention: multi-head self-attention and the KV cache.
    transformer: the IDiomTransformer and its blocks.
    sampling: KV-cached autoregressive generation.
    activations: residual-stream activation extraction.
    extract: sequence and FASTA embedding, and the idiom_extract CLI.
    io: loading a model from a checkpoint, a released directory, or the Hub.
"""

from idiom.model.attention import KVCache
from idiom.model.config import ModelConfig, idiom_20m, idiom_85m, idiom_300m
from idiom.model.sampling import generate, sample_next_token
from idiom.model.transformer import IDiomTransformer

__all__ = [
    "IDiomTransformer",
    "KVCache",
    "ModelConfig",
    "generate",
    "idiom_20m",
    "idiom_85m",
    "idiom_300m",
    "sample_next_token",
]
