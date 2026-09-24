"""SAE releases: sae_config.json and sae.safetensors.

The config records architecture, host model, layer, region, and FIM mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from idiom.data.fim import normalize_mode
from idiom.sae.model.sparse_coder import SparseCoder

SAE_CONFIG_FILE = "sae_config.json"
SAE_WEIGHTS_FILE = "sae.safetensors"


def save_sae(
    sae: SparseCoder,
    out_dir: str | Path,
    *,
    host_model: str | None,
    layer: int,
    region: str = "all",
    fim_mode: str = "prompted",
) -> Path:
    """Write sae_config.json and sae.safetensors for a trained SAE.

    Args:
        sae: The sparse coder to serialize.
        out_dir: Directory to write the release into; created if needed.
        host_model: The checkpoint path or Hub repo id of the host model, recorded so the release
            can load its host.
        layer: Zero-based training block index.
        region: The residue region the SAE was trained on: "all", "idr", or "non_idr".
        fim_mode: The prompt format the residual stream was taken under: "prompted" or "unprompted".

    Returns:
        The output directory.

    Raises:
        ValueError: If fim_mode is neither "prompted" nor "unprompted".
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "host_model": str(host_model) if host_model is not None else None,
        "layer": int(layer),
        "region": str(region),
        "fim_mode": normalize_mode(fim_mode),
        "d_in": int(sae.d_in),
        "num_latents": int(sae.num_latents),
        "k": int(sae.k.item()),
        "expansion_factor": int(sae.num_latents // sae.d_in),
        "activation": sae.activation,
        "multi_topk": bool(sae.multi_topk),
    }
    (d / SAE_CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
    # Hugging Face counts downloads of config.json; SAE loading still uses sae_config.json.
    (d / "config.json").write_bytes((d / SAE_CONFIG_FILE).read_bytes())
    save_model(sae, str(d / SAE_WEIGHTS_FILE))
    return d


def load_sae(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[SparseCoder, dict]:
    """Load a released SAE directory into a SparseCoder and its config.

    Args:
        path: Directory holding sae_config.json and sae.safetensors.
        device: Device to move the loaded model onto.

    Returns:
        The SparseCoder in eval mode, and the config dict, which carries host_model, layer, region,
        and fim_mode.

    Raises:
        ValueError: If the config records a fim_mode that is not a valid prompting mode.
    """
    d = Path(path)
    cfg = json.loads((d / SAE_CONFIG_FILE).read_text())
    if "fim_mode" in cfg:
        cfg["fim_mode"] = normalize_mode(cfg["fim_mode"])
    sae = SparseCoder(
        cfg["d_in"],
        num_latents=cfg["num_latents"],
        k=cfg["k"],
        activation=cfg.get("activation", "topk"),
        multi_topk=cfg.get("multi_topk", False),
    )
    load_model(sae, str(d / SAE_WEIGHTS_FILE))
    return sae.to(device).eval(), cfg
