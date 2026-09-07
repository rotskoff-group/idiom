# /// script
# requires-python = ">=3.10"
# dependencies = ["finches @ git+https://github.com/idptools/finches.git"]
# ///
"""Score FINCHES interaction epsilon (https://github.com/idptools/finches).

Use --mode homotypic for self-interaction or --mode heterotypic --partner SEQUENCE.
Choose --forcefield mpipi or calvados. Negative epsilon indicates attraction.
Run with uv run --script; see cookbook/rewards/README.md for reward configuration.
"""

import argparse
import json
import os
import sys


def build_frontend(forcefield):
    """Return the FINCHES frontend for a force field.

    Args:
        forcefield (str): "mpipi" or "calvados".

    Returns:
        object: The frontend, which exposes epsilon(seq1, seq2).
    """
    if forcefield == "calvados":
        from finches import CALVADOS_frontend
        return CALVADOS_frontend()
    from finches import Mpipi_frontend
    return Mpipi_frontend()


def build():
    """Parse arguments and return the epsilon scorer for the chosen mode and force field."""
    ap = argparse.ArgumentParser(description="FINCHES epsilon as an IDiom external reward.")
    ap.add_argument("--mode", default="homotypic", choices=("homotypic", "heterotypic"),
                    help="self-interaction, or interaction with --partner")
    ap.add_argument("--partner", default=None,
                    help="partner residue string, required for --mode heterotypic")
    ap.add_argument("--forcefield", default="mpipi", choices=("mpipi", "calvados"),
                    help="coarse-grained force field epsilon is derived from")
    args = ap.parse_args()
    if args.mode == "heterotypic" and not args.partner:
        raise SystemExit("--mode heterotypic needs --partner <sequence>")

    frontend = build_frontend(args.forcefield)
    partner = args.partner if args.mode == "heterotypic" else None  # None -> self-interaction

    def score_batch(sequences):
        """Return epsilon(seq, partner-or-self) for each sequence."""
        return [float(frontend.epsilon(s, partner or s)) for s in sequences]

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
