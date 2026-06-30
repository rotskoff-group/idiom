"""idiom.sae — top-k sparse autoencoders for interpreting IDiom.

An SAE is defined by the slice of a host model's residual stream it reads: ``(host_model, layer,
region)``. That triple is recorded in the release (:mod:`idiom.sae.io`) and applied identically at
every stage, so the autoencoder is always used on the distribution it was trained on.

Layout:
- :mod:`idiom.sae.sparse_coder` — the top-k ``SparseCoder`` model;
- :mod:`idiom.sae.io` — the release format (``sae_config.json`` + ``sae.safetensors``);
- :mod:`idiom.sae.training` — streaming ``ActivationStore`` + ``LitSAE`` + the ``idiom_sae`` entrypoint;
- :mod:`idiom.sae.features` — the per-residue feature-activation dataset, its reader, and the viewer;
- :mod:`idiom.sae.steering` — residual-stream hooks and feature-steered generation;
- :mod:`idiom.sae.eval` — held-out validation: substitution-loss "fraction recovered",
  reconstruction (FVU), sparsity/density, and the ``run_sae_eval`` entrypoint.
"""

from idiom.sae.io import load_sae, save_sae
from idiom.sae.sparse_coder import SparseCoder

__all__ = ["SparseCoder", "load_sae", "save_sae"]
