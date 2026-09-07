# /// script
# requires-python = ">=3.10"
# dependencies = ["biopython>=1.83", "numpy<2"]
# ///
"""Template for a scorer with isolated dependencies; copy and adapt.

Run with uv run --script to install the PEP 723 dependencies. This example scores
Biopython isoelectric point or molecular weight, selected by --property.

Implement build() to load the model and return a batch scorer. Keep serve() unchanged:
stdin accepts {"sequences": [...]}; stdout returns {"scores": [...]} or {"error": "..."},
one JSON object per line. Scores must be finite and match input order.
See cookbook/rewards/README.md for configuration and command checks.
"""

import argparse
import json
import os
import sys


def build():
    """Load dependencies once and return a batch scorer; serve() redirects library output to stderr."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="isoelectric_point",
                    choices=["isoelectric_point", "molecular_weight"])
    prop = ap.parse_args().property

    from Bio.SeqUtils.ProtParam import ProteinAnalysis  # a heavy import belongs here, not at top

    def score_batch(sequences):
        """Return one property value per non-empty sequence; let serve() report errors."""
        out = []
        for seq in sequences:
            analysis = ProteinAnalysis(seq)
            out.append(analysis.isoelectric_point() if prop == "isoelectric_point"
                       else analysis.molecular_weight())
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
