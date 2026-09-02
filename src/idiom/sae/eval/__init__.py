"""Held-out metrics for a trained SAE.

Modules:
    fidelity: substitution-loss "loss recovered", measuring how much of the host model's
        next-token prediction survives when the residual stream is replaced by its SAE
        reconstruction.
    reconstruction: reconstruction quality (FVU, explained variance) and sparsity (mean L0, dead
        fraction, per-latent firing frequency) over held-out activations.
"""

from idiom.sae.eval.fidelity import FidelityResult, compute_fidelity
from idiom.sae.eval.reconstruction import ReconstructionStats, reconstruction_stats

__all__ = [
    "FidelityResult",
    "compute_fidelity",
    "ReconstructionStats",
    "reconstruction_stats",
]
