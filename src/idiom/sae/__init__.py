"""Sparse autoencoders over an IDiom residual stream.

An SAE is described by its host model, the layer it reads, the residue region it was trained on,
and the prompt format those activations were taken under. All four are recorded in the release and
reapplied by the downstream tools.

Subpackages:
    model: the top-k SparseCoder and the release format (sae_config.json + sae.safetensors).
    train: the streaming ActivationStore, LitSAE, and the idiom_train_sae entrypoint.
    features: the per-residue feature-activation dataset, its reader, enrichment, and the viewer.
    steer: residual-stream hooks and feature-steered generation.
"""

from idiom.sae.model import SparseCoder, load_sae, save_sae

__all__ = ["SparseCoder", "load_sae", "save_sae"]
