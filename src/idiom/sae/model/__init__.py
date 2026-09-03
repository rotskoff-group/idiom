"""The SAE itself: the sparse coder and its release format.

Modules:
    sparse_coder: the top-k SparseCoder module.
    io: the release format, sae_config.json plus sae.safetensors.
"""

from idiom.sae.model.io import load_sae, save_sae
from idiom.sae.model.sparse_coder import SparseCoder

__all__ = ["SparseCoder", "load_sae", "save_sae"]
