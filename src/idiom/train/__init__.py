"""Training layer: one subpackage per training method, each self-contained.

Subpackages:
    autoreg: pretraining and SFT (the idiom_train_autoreg entrypoint).
    grpo: GRPO post-training against a composed reward (the idiom_train_grpo entrypoint).

Each holds its own LightningModule, its data handling where it differs, and its entrypoint.
"""

from idiom.train.autoreg import LitAutoregressive, warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
