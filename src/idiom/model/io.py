"""Load a pretrained IDiomTransformer from any saved form.

Two on-disk forms exist and this module reads both:

- a Lightning .ckpt: training output; a state_dict with a "model." prefix plus a
  hyper_parameters["model_cfg"] dict carrying the ModelConfig (every IDiom training module
  persists it, so checkpoints are self-describing).
- a released directory: config.json plus model.safetensors (see idiom.IDiom).

The architecture is never re-declared downstream; it is always recovered from the artifact.
config_from_checkpoint reads it from a .ckpt, load_pretrained loads a ckpt, and load_model is
format-agnostic and returns (model, cfg) for either form.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from idiom.model.config import ModelConfig
from idiom.model.transformer import IDiomTransformer

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def _load_checkpoint(ckpt_path: str | Path) -> tuple[ModelConfig, dict]:
    """Read a Lightning ckpt once, returning its stored ModelConfig and state_dict.

    Args:
        ckpt_path (str | Path): Path to the Lightning checkpoint.

    Returns:
        tuple[ModelConfig, dict]: The architecture config and the raw state_dict.

    Raises:
        ValueError: If the checkpoint carries no stored ModelConfig.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = (obj.get("hyper_parameters") or {}).get("model_cfg") if isinstance(obj, dict) else None
    if not cfg_dict:
        raise ValueError(
            f"{ckpt_path} carries no stored ModelConfig — not an IDiom training checkpoint."
        )
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    return ModelConfig(**cfg_dict), sd


def config_from_checkpoint(ckpt_path: str | Path) -> ModelConfig:
    """Recover the ModelConfig stored in a Lightning ckpt's hyperparameters.

    Args:
        ckpt_path (str | Path): Path to the Lightning checkpoint.

    Returns:
        ModelConfig: The architecture config stored in the checkpoint.

    Raises:
        ValueError: If the checkpoint carries no stored ModelConfig.
    """
    return _load_checkpoint(ckpt_path)[0]


def load_pretrained(
    ckpt_path: str | Path, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> tuple[IDiomTransformer, ModelConfig]:
    """Load an IDiomTransformer from a Lightning .ckpt (architecture read from the ckpt).

    Args:
        ckpt_path (str | Path): Path to the Lightning checkpoint.
        device (str | torch.device): Device to move the model to.
        eval_mode (bool): If True, put the model in eval mode before returning.

    Returns:
        tuple[IDiomTransformer, ModelConfig]: The loaded model and its config.
    """
    cfg, sd = _load_checkpoint(ckpt_path)
    model = IDiomTransformer(cfg)
    # keep only the policy weights ("model." prefix); ignores e.g. GRPO's frozen "reference.*"
    model.load_state_dict({k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")})
    if eval_mode:
        model.eval()
    return model.to(device), cfg


def load_released(
    path: str | Path, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> tuple[IDiomTransformer, ModelConfig]:
    """Load a released config.json plus model.safetensors directory into (model, cfg).

    Args:
        path (str | Path): Path to the released model directory.
        device (str | torch.device): Device to move the model to.
        eval_mode (bool): If True, put the model in eval mode before returning.

    Returns:
        tuple[IDiomTransformer, ModelConfig]: The loaded model and its config.
    """
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
    """Load a model from either on-disk form, returning (model, cfg).

    A directory with a config.json is a released model; anything else is a Lightning .ckpt. This
    is the single load entry point for every downstream stage.

    Args:
        path (str | Path): Path to a released model directory or a Lightning checkpoint.
        device (str | torch.device): Device to move the model to.
        eval_mode (bool): If True, put the model in eval mode before returning.

    Returns:
        tuple[IDiomTransformer, ModelConfig]: The loaded model and its config.
    """
    p = Path(path)
    if p.is_dir() and (p / CONFIG_FILE).exists():
        return load_released(p, device=device, eval_mode=eval_mode)
    return load_pretrained(p, device=device, eval_mode=eval_mode)
