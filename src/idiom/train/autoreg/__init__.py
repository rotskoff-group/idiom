"""Autoregressive pretraining and completion-masked fine-tuning."""

from idiom.train.autoreg.lit_autoreg import LitAutoregressive
from idiom.train.autoreg.schedulers import warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
