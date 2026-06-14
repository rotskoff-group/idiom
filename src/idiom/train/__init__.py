"""idiom.train — pretraining and post-training (GRPO) on PyTorch Lightning.

v2: autoregressive pretraining LightningModule with a custom warmup-cosine schedule
(no pl_bolts), and a GRPO/ProtGPS post-training path whose online generation uses the
KV-cached sampler from ``idiom.model``.

`LitAutoregressive` (shared by pretrain + SFT) + warmup-cosine schedule, and the GRPO
post-training path (`train/grpo/`).
"""

from idiom.train.lit_autoregressive import LitAutoregressive
from idiom.train.schedulers import warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
