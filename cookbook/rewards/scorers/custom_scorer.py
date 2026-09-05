# /// script
# requires-python = ">=3.10"
# dependencies = ["biopython>=1.83", "numpy<2"]
# ///
"""Template for a reward model in its own environment. Copy this file and edit it.

Use this only when the scorer cannot be imported into IDiom's environment; when it can, a term
naming an in-process function is simpler and faster -- see custom_rewards.py. This one computes
isoelectric point and molecular weight from Biopython, and pins `numpy<2` to stand in for a real
dependency conflict.

A scorer is a standalone program importing nothing from IDiom, so it can be a uv script (as here),
a conda environment, or `docker run -i`. Its dependencies live in the PEP 723 header above, so `uv
run --script` builds and caches the environment on first use.

    reward.terms:
      - {reward: {name: scorer,
                  cmd: "uv run --script /path/to/custom_scorer.py --property isoelectric_point"},
         shaping: {name: gaussian, target: 4.5, width: 0.3}, label: pI, weight: 1.0}

You write build(), which loads the model once and returns score_batch; serve() (pasted verbatim at
the bottom) drives the protocol. The protocol is newline-delimited JSON, one exchange per GRPO step:

    ->  {"sequences": ["ACDEF...", "GHIKL..."]}
    <-  {"scores": [4.21, 9.87]}          # or {"error": "..."}

One finite score per sequence, in the order they were sent. serve() drops empty completions (they
score 0.0), claims stdout for the protocol, and drops this file's directory from sys.path -- so a
library that prints on import cannot corrupt a response, and a scorer named after the package it
wraps does not shadow it. Copy serve() into your own scorer unchanged.

Check a command before running it -- this builds the environment, runs the handshake, and prints
what the shaping does to the raw value:

    uv run python -m idiom.train.grpo.reward.external \
        --cmd "uv run --script cookbook/rewards/scorers/custom_scorer.py --property isoelectric_point" \
        --shaping gaussian --target 4.5 --width 0.3
"""

import argparse
import json
import os
import sys


def build():
    """Parse arguments, load the model once, and return score_batch.

    Everything expensive -- argument parsing, heavy imports, loading weights -- happens here, once.
    serve() calls it after claiming stdout for the protocol, so a library that chatters on import
    (TensorFlow, ProtGPS, STARLING) writes to the log, not to the response stream.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="isoelectric_point",
                    choices=["isoelectric_point", "molecular_weight"])
    prop = ap.parse_args().property

    from Bio.SeqUtils.ProtParam import ProteinAnalysis  # a heavy import belongs here, not at top

    def score_batch(sequences):
        """Return the requested property for each sequence.

        serve() has already dropped empty completions, so every sequence here is non-empty and the
        returned list lines up with it one-to-one. Raise on a bad sequence rather than returning a
        sentinel: serve() turns the exception into an {"error": ...} response and the run surfaces
        it, instead of a wrong number silently steering training.
        """
        out = []
        for seq in sequences:
            analysis = ProteinAnalysis(seq)
            out.append(analysis.isoelectric_point() if prop == "isoelectric_point"
                       else analysis.molecular_weight())
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
