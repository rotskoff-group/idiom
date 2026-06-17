"""Training the SAE: streaming activations + the Lightning module + the ``idiom_sae`` entrypoint."""

from idiom.sae.training.activation_store import ActivationStore
from idiom.sae.training.lit_sae import LitSAE

__all__ = ["ActivationStore", "LitSAE"]
