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
IDIOM_PROTGPS_DEVICE selects the device (default CUDA if available, otherwise CPU).
PROTGPS_BATCH sets sequences per forward pass (default 32). Checkpoints download on first use.
"""

import argparse
import json
import os
import pickle
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import torch

COMPARTMENTS = [
    "nuclear_speckle", "p-body", "pml-bdoy", "post_synaptic_density", "stress_granule",
    "chromosome", "nucleolus", "nuclear_pore_complex", "cajal_body", "rna_granule",
    "cell_junction", "transcriptional",
]

ZENODO_URL = "https://zenodo.org/records/14795445/files/checkpoints.zip?download=1"
_CKPT_STEM = "protgps/32bf44b16a4e770a674896b81dfb3729"
_MAX_LEN = 1800  # ProtGPS sequence-length ceiling
_BATCH = int(os.environ.get("PROTGPS_BATCH", "32"))


def _device():
    """Return the torch device for the classifier."""
    return os.environ.get("IDIOM_PROTGPS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_dir() -> Path:
    """Return the checkpoint directory, downloading the Zenodo release on first use.

    Returns:
        A directory holding <stem>.args and <stem>epoch=26.ckpt.
    """
    d = Path(os.environ.get("IDIOM_PROTGPS_DIR",
                            Path.home() / ".cache/idiom/protgps")).expanduser()
    # the Zenodo archive unpacks as checkpoints/protgps/..., so accept either layout
    for cand in (d, d / "checkpoints"):
        if (cand / f"{_CKPT_STEM}.args").exists():
            return cand
    import requests  # only needed on the first run

    d.mkdir(parents=True, exist_ok=True)
    zip_path = d / "checkpoints.zip"
    print(f"downloading ProtGPS checkpoints (166 MB, CC BY 4.0) to {d} ...", file=sys.stderr,
          flush=True)
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
    raise SystemExit(f"the Zenodo archive did not contain {_CKPT_STEM}.args under {d}; "
                     f"set IDIOM_PROTGPS_DIR to a directory holding the ProtGPS checkpoints")


def _load_model():
    """Load the ProtGPS classifier, downloading its checkpoints if needed.

    Returns:
        The ProtGPS lightning module, in eval mode on the chosen device.
    """
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


def build():
    """Parse --compartment, load the classifier, and return the compartment-probability scorer."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compartment", default="nucleolus",
                    help="one of the 12 compartments, or max / mean over them")
    target = ap.parse_args().compartment
    if target not in COMPARTMENTS and target not in ("max", "mean"):
        raise SystemExit(f"--compartment {target!r} is not one of {COMPARTMENTS} (or max, mean)")

    model = _load_model()

    @torch.no_grad()
    def score_batch(sequences):
        """Return the compartment probability for each sequence."""
        scores = [0.0] * len(sequences)
        for start in range(0, len(sequences), _BATCH):
            chunk = sequences[start:start + _BATCH]
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


def serve(build):
    """Serve newline-delimited JSON requests until stdin closes.

    Call build() once to obtain a batch scorer. Redirect library output to stderr,
    score empty sequences as 0, and report scoring exceptions as JSON errors.
    """
    # This file's own directory is sys.path[0]; drop it so a scorer named after the package it wraps
    # (sparrow.py importing sparrow) resolves to the installed package, not back to itself.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    # stdout is the protocol. A library that prints on import (TensorFlow, ProtGPS, STARLING) would
    # corrupt the first response, so keep the real stdout for responses and send chatter to stderr.
    out, sys.stdout = sys.stdout, sys.stderr
    score_batch = build()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            seqs = json.loads(line)["sequences"]
            keep = [(i, s) for i, s in enumerate(seqs) if s]  # empty completions score 0.0
            values = score_batch([s for _, s in keep]) if keep else []
            scores = [0.0] * len(seqs)
            for (i, _), v in zip(keep, values):
                scores[i] = float(v)
            payload = {"scores": scores}
        except Exception as e:
            payload = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(payload), file=out, flush=True)


if __name__ == "__main__":
    serve(build)
