"""Autoregressive training: pretraining from scratch and supervised fine-tuning.

Pretraining and SFT share one LightningModule; the only difference is the loss mask, so SFT is a
config, not a separate code path. This package sits alongside grpo so a further post-training
method (DPO, say) is added as a sibling rather than by editing what is here.

Modules:
    lit_autoreg: the LightningModule shared by pretraining and SFT.
    schedulers: the warmup-cosine learning-rate schedule.
    train_autoreg: the idiom_train entrypoint.
"""

from idiom.train.autoreg.lit_autoreg import LitAutoregressive
from idiom.train.autoreg.schedulers import warmup_cosine

__all__ = ["LitAutoregressive", "warmup_cosine"]
