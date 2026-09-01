"""Pretraining and post-training (GRPO) on PyTorch Lightning.

Provides an autoregressive pretraining LightningModule with a custom warmup-cosine schedule and
a GRPO/ProtGPS post-training path whose online generation uses the KV-cached sampler from
idiom.model. LitAutoregressive is shared by pretraining and SFT; the GRPO post-training path
lives in train/grpo/.
"""

from idiom.train.lit_autoregressive import LitAutoregressive
from idiom.train.schedulers import warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
