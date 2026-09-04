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
"""Score IDRs with ProtGPS (https://github.com/pgmikhael/protgps) as an external GRPO reward.

ProtGPS is a condensate-localization classifier over a small ESM-2, mapping a residue string to 12
compartment probabilities. It is the second worked example of an external reward, and the one that
shows why the mechanism exists: its environment pins python 3.8, torch 2.0 and
pytorch-lightning 1.6.4, none of which can coexist with IDiom (python >= 3.10, torch >= 2.4). The
PEP 723 header above is that whole environment, built and cached by uv on first use.

The checkpoints (166 MB, CC BY 4.0) are downloaded once from the paper's Zenodo record into
IDIOM_PROTGPS_DIR, so a clean checkout needs no manual setup:

    uv run python -m idiom.train.grpo.reward.external \
        --cmd "uv run --script cookbook/rewards/scorers/protgps.py --compartment nucleolus"

In configs/grpo.yaml:

    reward.terms:
      - {cmd: "uv run --script cookbook/rewards/scorers/protgps.py --compartment nucleolus",
         label: protgps, weight: 1.0}

The raw reward is already a probability in [0, 1], so a term usually leaves it unshaped.
--compartment is one of the 12 compartments below, or "max" / "mean" over them.

Caveat: ProtGPS rewards compositional extremity, and high scores are reachable with low-complexity
tracts, so keep the entropy and length guardrails on. Optimizing a classifier is not the same as
reproducing what it detects -- that gap is what the SAE feature reward (sae_feature) addresses.

Environment variables:
    IDIOM_PROTGPS_DIR     where the checkpoints live; downloaded here on first use
                          (default ~/.cache/idiom/protgps).
    IDIOM_PROTGPS_DEVICE  torch device (default cuda if available else cpu).
    PROTGPS_BATCH         sequences per forward pass (default 32).
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

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--compartment", default="nucleolus",
                 help="one of the 12 compartments, or max / mean over them")
TARGET = _ap.parse_args().compartment
if TARGET not in COMPARTMENTS and TARGET not in ("max", "mean"):
    raise SystemExit(f"--compartment {TARGET!r} is not one of {COMPARTMENTS} (or max, mean)")


# ProtGPS prints to stdout while loading ("Using ESM hidden layers 6"), and stdout is the protocol:
# any stray line there is read as a malformed response. Keep the real stdout for responses only and
# send everything else to stderr, which the parent forwards to its log. Any scorer wrapping a
# library that prints needs this.
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def _device():
    """Return the torch device for the classifier."""
    return os.environ.get("IDIOM_PROTGPS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_dir() -> Path:
    """Return the checkpoint directory, downloading the Zenodo release on first use.

    Returns:
        Path: A directory holding <stem>.args and <stem>epoch=26.ckpt.
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
    """Load the ProtGPS classifier once, downloading its checkpoints if needed.

    Returns:
        The ProtGPS lightning module, in eval mode on the chosen device.
    """
    # This file is named protgps.py, and python puts a script's own directory on sys.path[0]; drop
    # it so "import protgps" resolves to the installed package rather than back to this file.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
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


MODEL = _load_model()  # once per process, not once per batch


@torch.no_grad()
def score_batch(sequences):
    """Score a batch of sequences for the configured compartment.

    Args:
        sequences (list[str]): IDR residue strings.

    Returns:
        list[float]: The compartment probability for each sequence, 0.0 for empty strings.
    """
    keep = [(i, s[:_MAX_LEN]) for i, s in enumerate(sequences) if s]
    scores = [0.0] * len(sequences)
    for start in range(0, len(keep), _BATCH):
        chunk = keep[start:start + _BATCH]
        probs = torch.sigmoid(MODEL.model({"x": [s for _, s in chunk]})["logit"]).cpu()
        for (i, _), row in zip(chunk, probs):
            if TARGET == "max":
                scores[i] = float(row.max())
            elif TARGET == "mean":
                scores[i] = float(row.mean())
            else:
                scores[i] = float(row[COMPARTMENTS.index(TARGET)])
    return scores


def main():
    """Read batches from stdin and write scores to stdout until the parent closes the pipe."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = {"scores": score_batch(json.loads(line)["sequences"])}
        except Exception as e:
            response = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(response), file=_PROTOCOL_STDOUT, flush=True)


if __name__ == "__main__":
    main()
