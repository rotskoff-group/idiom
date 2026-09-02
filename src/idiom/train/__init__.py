"""Training layer: pretraining, SFT, and GRPO post-training on PyTorch Lightning.

Modules:
    lit_autoregressive: the LightningModule shared by pretraining and SFT.
    schedulers: the warmup-cosine learning-rate schedule.
    train: the idiom_train entrypoint.
    grpo: the GRPO post-training package.
"""

from idiom.train.lit_autoregressive import LitAutoregressive
from idiom.train.schedulers import warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
