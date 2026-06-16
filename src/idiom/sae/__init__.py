"""idiom.sae — top-k sparse autoencoders for interpreting IDiom (was ``idiomatics``).

v2: the ``SparseCoder`` / ``LitSAE`` port plus a streaming ``ActivationStore`` that runs
the frozen IDiom model live during SAE training (residue-only positions, no activation
h5). Fidelity and steering hook the model directly — the old ``idiom_interface`` bridge
is gone.

Status: P5 — ``sparse_coder`` / ``lit_sae`` + streaming ``activation_store`` (no h5), and
``fidelity`` / ``steering`` rewired onto :class:`IDiomTransformer` (hooks on ``model.blocks``,
residue masking via ``Tokenizer.residue_mask``, steered generation via the KV-cached sampler).
See REFACTOR_PLAN.md (P5).
"""

from idiom.sae.io import load_sae, save_sae
from idiom.sae.sparse_coder import SparseCoder, build_sae

__all__ = ["SparseCoder", "build_sae", "load_sae", "save_sae"]
