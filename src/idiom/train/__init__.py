"""idiom.train — pretraining and post-training (GRPO) on PyTorch Lightning.

v2: autoregressive pretraining LightningModule with a custom warmup-cosine schedule
(no pl_bolts), and a GRPO/ProtGPS post-training path whose online generation uses the
KV-cached sampler from ``idiom.model``.

Status: P3 — `LitAutoregressive` (shared by pretrain + SFT) + warmup-cosine schedule landed.
GRPO (P4) and the Hydra entrypoint/configs are next. Migration source: legacy
``idiom.nn.transformer.{module,losses}``.
"""

from idiom.train.lit_autoregressive import LitAutoregressive
from idiom.train.schedulers import warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
