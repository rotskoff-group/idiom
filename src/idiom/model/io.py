"""Load a pretrained :class:`IDiomTransformer` from any saved form.

Two on-disk forms exist and this module reads both:

* a **Lightning ``.ckpt``** — training output; a ``state_dict`` with a ``model.`` prefix plus a
  ``hyper_parameters["model_cfg"]`` dict carrying the :class:`ModelConfig` (every IDiom training
  module persists it, so checkpoints are self-describing).
* a **released directory** — ``config.json`` + ``model.safetensors`` (see :meth:`idiom.IDiom`).

The architecture is never re-declared downstream: it is always recovered from the artifact.
``config_from_checkpoint`` reads it from a ``.ckpt``; ``load_pretrained`` loads a ckpt; ``load_model``
is format-agnostic and returns ``(model, cfg)`` for either form.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from idiom.model.config import ModelConfig
from idiom.model.transformer import IDiomTransformer

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def config_from_checkpoint(ckpt_path: str | Path) -> ModelConfig:
    """Recover the :class:`ModelConfig` stored in a Lightning ckpt's hyperparameters."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = (obj.get("hyper_parameters") or {}).get("model_cfg") if isinstance(obj, dict) else None
    if not cfg_dict:
        raise ValueError(
            f"{ckpt_path} carries no stored ModelConfig — not an IDiom training checkpoint."
        )
    return ModelConfig(**cfg_dict)


def load_pretrained(
    ckpt_path: str | Path, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> IDiomTransformer:
    """Load an :class:`IDiomTransformer` from a Lightning ``.ckpt`` (arch read from the ckpt)."""
    cfg = config_from_checkpoint(ckpt_path)
    model = IDiomTransformer(cfg)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    # keep only the policy weights ("model." prefix); ignores e.g. GRPO's frozen "reference.*"
    model.load_state_dict({k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")})
    if eval_mode:
        model.eval()
    return model.to(device)


def load_released(
    path: str | Path, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> tuple[IDiomTransformer, ModelConfig]:
    """Load a released ``config.json`` + ``model.safetensors`` directory → ``(model, cfg)``."""
    from safetensors.torch import load_model  # noqa: PLC0415

    d = Path(path)
    cfg = ModelConfig(**json.loads((d / CONFIG_FILE).read_text()))
    model = IDiomTransformer(cfg)
    load_model(model, str(d / WEIGHTS_FILE))  # handles the tied embedding
    if eval_mode:
        model.eval()
    return model.to(device), cfg


def load_model(
    path: str | Path, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> tuple[IDiomTransformer, ModelConfig]:
    """Format-agnostic loader → ``(model, cfg)``.

    A directory with ``config.json`` is a released model; anything else is a Lightning ``.ckpt``.
    The single load entry point for every downstream stage.
    """
    p = Path(path)
    if p.is_dir() and (p / CONFIG_FILE).exists():
        return load_released(p, device=device, eval_mode=eval_mode)
    return load_pretrained(p, device=device, eval_mode=eval_mode), config_from_checkpoint(p)
