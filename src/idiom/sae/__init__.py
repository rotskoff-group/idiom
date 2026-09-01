"""idiom.sae — top-k sparse autoencoders for interpreting IDiom.

An SAE is defined by the slice of a host model's residual stream it reads: (host_model, layer,
region). That triple is recorded in the release (idiom.sae.io) and applied identically at every
stage, so the autoencoder is always used on the distribution it was trained on.

Layout:
- idiom.sae.sparse_coder — the top-k SparseCoder model;
- idiom.sae.io — the release format (sae_config.json + sae.safetensors);
- idiom.sae.training — streaming ActivationStore + LitSAE + the idiom_sae entrypoint;
- idiom.sae.features — the per-residue feature-activation dataset, its reader, and the viewer;
- idiom.sae.steering — residual-stream hooks and feature-steered generation;
- idiom.sae.eval — held-out validation: substitution-loss "fraction recovered", reconstruction
  (FVU), and sparsity/density.
"""

from idiom.sae.io import load_sae, save_sae
from idiom.sae.sparse_coder import SparseCoder

__all__ = ["SparseCoder", "load_sae", "save_sae"]
