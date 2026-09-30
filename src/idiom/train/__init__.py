"""Autoregressive pretraining, supervised fine-tuning, and GRPO post-training."""

from idiom.train.autoreg import LitAutoregressive, warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
