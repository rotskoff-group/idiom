# /// script
# requires-python = "==3.8.*"
# dependencies = [
#   "protgps @ git+https://github.com/pgmikhael/protgps",
#   "torch==2.0.0",
#   "pytorch-lightning==1.6.4",
#   "fair-esm",
#   "numpy==1.23.4",
#   "pandas==2.0.3",
#   "requests",
# ]
# ///
"""Score ProtGPS compartment probabilities (https://github.com/pgmikhael/protgps).

Run with uv run --script to use the isolated dependencies in the PEP 723 header.
--compartment selects one of 12 compartments, or "max" / "mean" across them.
IDIOM_PROTGPS_DIR sets the checkpoint cache (default ~/.cache/idiom/protgps);
IDIOM_PROTGPS_DEVICE selects the device (default CPU for compatibility with the pinned torch).
PROTGPS_BATCH sets sequences per forward pass (default 1 for batch-independent rewards).
Checkpoints download on first use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule

import argparse
import os
import pickle
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

from _scorer_protocol import serve

COMPARTMENTS = [
    "nuclear_speckle",
    "p-body",
    "pml-bdoy",
    "post_synaptic_density",
    "stress_granule",
    "chromosome",
    "nucleolus",
    "nuclear_pore_complex",
    "cajal_body",
    "rna_granule",
    "cell_junction",
    "transcriptional",
]

ZENODO_URL = "https://zenodo.org/records/14795445/files/checkpoints.zip?download=1"
_CKPT_STEM = "protgps/32bf44b16a4e770a674896b81dfb3729"
_MAX_LEN = 1800  # ProtGPS sequence-length ceiling
_BATCH = int(os.environ.get("PROTGPS_BATCH", "1"))


def _device() -> str:
    """Return IDIOM_PROTGPS_DEVICE when set, otherwise CPU."""
    return os.environ.get("IDIOM_PROTGPS_DEVICE") or "cpu"


def _checkpoint_dir() -> Path:
    """Download the Zenodo checkpoint release if needed and return its directory."""
    d = Path(os.environ.get("IDIOM_PROTGPS_DIR", Path.home() / ".cache/idiom/protgps")).expanduser()
    # the Zenodo archive unpacks as checkpoints/protgps/..., so accept either layout
    for cand in (d, d / "checkpoints"):
        if (cand / f"{_CKPT_STEM}.args").exists():
            return cand
    import requests

    d.mkdir(parents=True, exist_ok=True)
    zip_path = d / "checkpoints.zip"
    print(f"downloading ProtGPS checkpoints (166 MB, CC BY 4.0) to {d} ...", file=sys.stderr, flush=True)
    with requests.get(ZENODO_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(d)
    zip_path.unlink()
    for cand in (d, d / "checkpoints"):
        if (cand / f"{_CKPT_STEM}.args").exists():
            return cand
    raise SystemExit(
        f"the Zenodo archive did not contain {_CKPT_STEM}.args under {d}; "
        f"set IDIOM_PROTGPS_DIR to a directory holding the ProtGPS checkpoints"
    )


def _load_model() -> LightningModule:
    """Load the ProtGPS classifier in eval mode on the selected device."""
    from protgps.utils.loading import get_object

    parent = _checkpoint_dir()
    args = Namespace(**pickle.load(open(parent / f"{_CKPT_STEM}.args", "rb")))
    args.model_path = str(parent / f"{_CKPT_STEM}epoch=26.ckpt")
    args.pretrained_hub_dir = str(parent / "esm_models/esm2")  # torch.hub cache for the backbone
    Path(args.pretrained_hub_dir).mkdir(parents=True, exist_ok=True)

    model = get_object(args.lightning_name, "lightning")(args)
    model = model.load_from_checkpoint(
        checkpoint_path=args.model_path,
        strict=not args.relax_checkpoint_matching,
        args=args,
    )
    return model.eval().to(_device())


def build() -> Callable[[list[str]], list[float]]:
    """Build a scorer for a named compartment or the max/mean across all compartments.

    Parse --compartment, load the classifier, and truncate each sequence to its first
    1,800 residues before prediction. Return one probability or aggregate per sequence.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--compartment", default="nucleolus", help="one of the 12 compartments, or max / mean over them"
    )
    target = ap.parse_args().compartment
    if target not in COMPARTMENTS and target not in ("max", "mean"):
        raise SystemExit(f"--compartment {target!r} is not one of {COMPARTMENTS} (or max, mean)")

    import torch

    model = _load_model()

    @torch.no_grad()
    def score_batch(sequences) -> list[float]:
        """Return compartment or aggregate probabilities for sequences truncated to 1,800 residues."""
        scores = [0.0] * len(sequences)
        for start in range(0, len(sequences), _BATCH):
            chunk = sequences[start : start + _BATCH]
            probs = torch.sigmoid(model.model({"x": [s[:_MAX_LEN] for s in chunk]})["logit"]).cpu()
            for j, row in enumerate(probs):
                if target == "max":
                    scores[start + j] = float(row.max())
                elif target == "mean":
                    scores[start + j] = float(row.mean())
                else:
                    scores[start + j] = float(row[COMPARTMENTS.index(target)])
        return scores

    return score_batch


if __name__ == "__main__":
    serve(build)
