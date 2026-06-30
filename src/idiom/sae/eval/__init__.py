"""idiom.sae.eval — held-out validation of a trained SAE (or a layer-swept set of them).

Everything here answers "is this SAE faithful?" on data it was *not* trained on, always on the
distribution it was trained on (the SAE's recorded ``region`` + ``fim_mode``):

- :mod:`idiom.sae.eval.fidelity` — downstream "loss recovered" (Gao): how much of IDiom's
  next-token prediction survives when the layer's residual stream is replaced by ``sae(residual)``.
  This is the gold-standard fidelity check (functional, not just MSE) and the residue-region
  edit-masking lives here.
- :mod:`idiom.sae.eval.reconstruction` — reconstruction (FVU / explained variance) and sparsity
  (mean L0, dead-feature fraction, per-latent firing frequency) over held-out activations.

These are reusable metric primitives (``IDiomSAE.fidelity`` builds on :func:`compute_fidelity`). The
operator CLI that sweeps a directory of SAE releases and writes the per-layer figure CSV lives in the
repo-only top-level ``eval/`` package (``python -m eval.run_sae_eval``), not in the shipped library.
"""

from idiom.sae.eval.fidelity import FidelityResult, compute_fidelity
from idiom.sae.eval.reconstruction import ReconstructionStats, reconstruction_stats

__all__ = [
    "FidelityResult",
    "compute_fidelity",
    "ReconstructionStats",
    "reconstruction_stats",
]
