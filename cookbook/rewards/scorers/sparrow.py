# /// script
# requires-python = ">=3.10"
# dependencies = ["sparrow @ git+https://github.com/idptools/sparrow.git"]
# ///
"""Score sparrow sequence properties (https://github.com/idptools/sparrow).

--property selects an ALBATROSS prediction or a Protein attribute such as FCR or NCPR.
Kappa returns -1 for sequences without charged residues.
Run with uv run --script; see cookbook/rewards/README.md for reward configuration.
"""

import argparse
import json
import os
import sys


def build():
    """Parse arguments and return the sparrow property scorer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="radius_of_gyration")
    prop = ap.parse_args().property
    from sparrow import Protein

    def score_batch(sequences):
        """Return the requested property for each sequence.

        ALBATROSS predictions are methods on Protein.predictor; sequence parameters such as FCR and
        kappa are attributes on Protein itself.
        """
        out = []
        for seq in sequences:
            protein = Protein(seq)
            predictor = getattr(protein.predictor, prop, None)
            out.append(float(predictor()) if callable(predictor) else float(getattr(protein, prop)))
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
