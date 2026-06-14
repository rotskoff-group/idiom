"""Load a pretrained :class:`IDiomTransformer` from a checkpoint.

Accepts a Lightning ``.ckpt`` (``state_dict`` with a ``model.`` prefix) or a bare model
``state_dict`` (e.g. released safetensors converted to a dict). The basis for the eventual
public ``IDiom.from_pretrained`` (HF download → this loader).
"""

from __future__ import annotations

from pathlib import Path

import torch

from idiom.model.config import ModelConfig
from idiom.model.transformer import IDiomTransformer


def load_pretrained(
    ckpt_path: str | Path, cfg: ModelConfig, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> IDiomTransformer:
    model = IDiomTransformer(cfg)
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    # strip the LightningModule's "model." prefix if present
    prefixed = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(prefixed or sd)
    if eval_mode:
        model.eval()
    return model.to(device)
