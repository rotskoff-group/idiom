"""Load models from Lightning checkpoints, release directories, or Hub repositories.

Checkpoints store model_cfg and "model."-prefixed weights; releases contain
config.json and model.safetensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

# aliased: this module defines its own load_model below
from safetensors.torch import load_model as _safetensors_load_model

from idiom.model.config import ModelConfig
from idiom.model.transformer import IDiomTransformer

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def _load_checkpoint(ckpt_path: str | Path) -> tuple[ModelConfig, dict]:
    """Read a Lightning checkpoint, returning its stored ModelConfig and state_dict.

    Args:
        ckpt_path: Path to the Lightning checkpoint.

    Returns:
        The architecture config and the raw state_dict.

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
    """Read ModelConfig from a Lightning checkpoint's hyperparameters.

    Raises:
        ValueError: If the checkpoint has no stored ModelConfig.
    """
    return _load_checkpoint(ckpt_path)[0]


def load_pretrained(
    ckpt_path: str | Path, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> tuple[IDiomTransformer, ModelConfig]:
    """Load an IDiomTransformer from a Lightning checkpoint.

    Only keys prefixed "model." are loaded; any other module in the checkpoint, such as GRPO's
    frozen reference policy, is ignored.

    Args:
        ckpt_path: Path to the Lightning checkpoint.
        device: Device to move the model to.
        eval_mode: If True, put the model in eval mode before returning.

    Returns:
        The loaded model and the config read from the checkpoint.

    Raises:
        ValueError: If the checkpoint carries no stored ModelConfig.
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
    """Load a released directory holding config.json and model.safetensors.

    Args:
        path: Path to the released model directory.
        device: Device to move the model to.
        eval_mode: If True, put the model in eval mode before returning.

    Returns:
        The loaded model and the config read from config.json.
    """
    d = Path(path)
    cfg = ModelConfig(**json.loads((d / CONFIG_FILE).read_text()))
    model = IDiomTransformer(cfg)
    _safetensors_load_model(model, str(d / WEIGHTS_FILE))  # handles the tied embedding
    if eval_mode:
        model.eval()
    return model.to(device), cfg


def load_model(
    path: str | Path, *, device: str | torch.device = "cpu", eval_mode: bool = True
) -> tuple[IDiomTransformer, ModelConfig]:
    """Load a model from a checkpoint, a released directory, or a Hub repo id.

    A path that does not exist locally is downloaded as a Hub repo id. A directory holding
    config.json is read as a released model; any other path is read as a Lightning checkpoint.

    Args:
        path: A Lightning checkpoint, a released model directory, or a Hub repo id.
        device: Device to move the model to.
        eval_mode: If True, put the model in eval mode before returning.

    Returns:
        The loaded model and its config.

    Raises:
        ValueError: If the artifact is a checkpoint that carries no stored ModelConfig.
    """
    p = Path(path)
    if not p.exists():  # not a local path: treat it as a HF repo id and fetch the released snapshot
        p = Path(snapshot_download(str(path)))
    if p.is_dir() and (p / CONFIG_FILE).exists():
        return load_released(p, device=device, eval_mode=eval_mode)
    return load_pretrained(p, device=device, eval_mode=eval_mode)
