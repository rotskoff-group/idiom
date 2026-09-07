# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = ["idptools-starling", "torch==2.4.1"]
# ///
"""Score STARLING ensemble dimensions (https://github.com/idptools/starling).

--property selects radius_of_gyration or end_to_end_distance; --conformations sets
ensemble size (default 20). STARLING_DEVICE overrides CUDA/CPU auto-selection.
Run with uv run --script. Ensemble generation is expensive; allow a generous scorer timeout.
The pinned torch build must match the CUDA driver.
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
