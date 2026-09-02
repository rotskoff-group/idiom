"""Sparse autoencoders over an IDiom residual stream.

An SAE is described by its host model, the layer it reads, the residue region it was trained on,
and the prompt format those activations were taken under. All four are recorded in the release and
reapplied by the downstream tools.

Subpackages and modules:
    sparse_coder: the top-k SparseCoder module.
    io: the release format, sae_config.json plus sae.safetensors.
    training: the streaming ActivationStore, LitSAE, and the idiom_sae entrypoint.
    features: the per-residue feature-activation dataset, its reader, enrichment, and the viewer.
    steering: residual-stream hooks and feature-steered generation.
    eval: substitution-loss fidelity, reconstruction, and sparsity metrics.
"""

from idiom.sae.io import load_sae, save_sae
from idiom.sae.sparse_coder import SparseCoder

__all__ = ["SparseCoder", "load_sae", "save_sae"]
