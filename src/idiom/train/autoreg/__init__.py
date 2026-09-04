"""Autoregressive training: pretraining from scratch and supervised fine-tuning.

Pretraining and SFT share one LightningModule and differ only in the loss mask, which is set by
config.

Modules:
    lit_autoreg: the LightningModule shared by pretraining and SFT.
    schedulers: the warmup-cosine learning-rate schedule.
    train_autoreg: the idiom_train_autoreg entrypoint.
"""

from idiom.train.autoreg.lit_autoreg import LitAutoregressive
from idiom.train.autoreg.schedulers import warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
