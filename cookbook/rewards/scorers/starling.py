# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = ["idptools-starling", "torch==2.4.1"]
# ///
"""Score IDRs with STARLING (https://github.com/idptools/starling) as an external GRPO reward.

STARLING is a diffusion model that generates coarse-grained conformational ensembles for an IDR, so
the reward is an ensemble average of the same dimensions sparrow's ALBATROSS predictors estimate
directly.

    --property radius_of_gyration | end_to_end_distance
    --conformations N             ensemble size per sequence (default 20)

    reward.terms:
      - {reward: {name: scorer, timeout: 900,
                  cmd: "uv run --script cookbook/rewards/scorers/starling.py --property radius_of_gyration"},
         shaping: {name: quadratic, target: 25, width: 0.2}, label: rg_ens, weight: 1.0}

Cost is about 9 s per 32 sequences at 20 conformations on an H100, roughly doubling GRPO step time,
so give the term a generous timeout. On CPU it is about 6 s per sequence, too slow to train
against.

The torch pin matters: the wheel idptools-starling resolves by default can be newer than the host
CUDA driver, which fails at model load with "The NVIDIA driver on your system is too old". Pin the
torch build that matches your driver, or run with CUDA_VISIBLE_DEVICES="" to fall back to CPU.

Environment variables:
    STARLING_DEVICE   torch device (default cuda when visible, else cpu).
"""

import argparse
import json
import os
import sys


def build():
    """Parse arguments and return the STARLING ensemble-dimension scorer."""
    ap = argparse.ArgumentParser(description="STARLING ensemble dimensions as an IDiom reward.")
    ap.add_argument("--property", default="radius_of_gyration",
                    choices=("radius_of_gyration", "end_to_end_distance"))
    ap.add_argument("--conformations", type=int, default=20)
    args = ap.parse_args()

    import torch
    from starling import generate

    device = os.environ.get("STARLING_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def score_batch(sequences):
        """Return the ensemble-average property per sequence."""
        keyed = {f"s{i}": s for i, s in enumerate(sequences)}
        ensembles = generate(keyed, conformations=args.conformations, device=device,
                             show_progress_bar=False)
        out = [0.0] * len(sequences)
        for key, ensemble in ensembles.items():
            out[int(key[1:])] = float(getattr(ensemble, args.property)(return_mean=True))
        return out

    return score_batch


def serve(build):
    """Drive the newline-JSON scorer protocol until stdin closes.

    build() is called once, after stdout is claimed for the protocol, and returns score_batch: a
    function mapping a list of (non-empty) residue strings to one raw score each. Doing the imports
    and model loading inside build() keeps any chatter they print off the protocol stream.

    Args:
        build (Callable[[], Callable[[list[str]], list[float]]]): Returns the batch scorer.
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
