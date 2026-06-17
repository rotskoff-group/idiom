"""Read/write a trained SAE in its release form.

An SAE only means something attached to a host model at a layer, so the release records both:
a ``sae_config.json`` (``host_model`` + ``layer`` + ``SparseCoder`` shape) and ``sae.safetensors``.
This is the single SAE artifact — training writes it, :class:`idiom.IDiomSAE` and the analysis
tools read it. Mirrors ``idiom.model.io`` for the transformer.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from idiom.sae.sparse_coder import SparseCoder

SAE_CONFIG_FILE = "sae_config.json"
SAE_WEIGHTS_FILE = "sae.safetensors"


def save_sae(
    sae: SparseCoder, out_dir: str | Path, *, host_model: str | None, layer: int, region: str = "all"
) -> Path:
    """Write ``sae_config.json`` + ``sae.safetensors``. ``host_model`` is the model checkpoint/repo
    the SAE was trained against (recorded so it can self-load its host); ``region`` is the slice of
    the residual stream it was trained on (``all`` | ``idr`` | ``non_idr``), so every downstream
    tool applies it to the same positions."""
    from safetensors.torch import save_model  # noqa: PLC0415

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "host_model": str(host_model) if host_model is not None else None,
        "layer": int(layer),
        "region": str(region),
        "d_in": int(sae.d_in),
        "num_latents": int(sae.num_latents),
        "k": int(sae.k.item()),
        "expansion_factor": int(sae.num_latents // sae.d_in),
        "activation": sae.activation,
        "multi_topk": bool(sae.multi_topk),
    }
    (d / SAE_CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
    save_model(sae, str(d / SAE_WEIGHTS_FILE))
    return d


def load_sae(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[SparseCoder, dict]:
    """Load a released SAE dir → ``(SparseCoder, config dict)`` (config carries host_model/layer)."""
    from safetensors.torch import load_model  # noqa: PLC0415

    d = Path(path)
    cfg = json.loads((d / SAE_CONFIG_FILE).read_text())
    sae = SparseCoder(
        cfg["d_in"], num_latents=cfg["num_latents"], k=cfg["k"],
        activation=cfg.get("activation", "topk"), multi_topk=cfg.get("multi_topk", False),
    )
    load_model(sae, str(d / SAE_WEIGHTS_FILE))
    return sae.to(device).eval(), cfg
