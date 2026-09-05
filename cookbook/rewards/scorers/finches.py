# /// script
# requires-python = ">=3.10"
# dependencies = ["finches @ git+https://github.com/idptools/finches.git"]
# ///
"""Score IDRs with FINCHES (https://github.com/idptools/finches) as an external GRPO reward.

FINCHES computes epsilon, a mean-field interaction parameter derived from a coarse-grained force
field (Mpipi-GG or CALVADOS). Negative epsilon means attractive, positive means repulsive.

Two modes:

    --mode homotypic                 epsilon(seq, seq): self-interaction, i.e. how strongly the
                                     sequence likes itself (LLPS propensity).
    --mode heterotypic --partner S   epsilon(seq, S): interaction with a fixed partner sequence,
                                     which is how you design a co-condensate or binding partner
                                     for a protein you already have.

Reference values on this scale (Mpipi, homotypic): an FUS-LC-like aromatic tract is about -8.5,
IDiom's base generations average +3.6 (sd 7.0), and natural ProtGPS nucleolus IDRs average +7.7.
Aim at a negative target to design self-attractive sequences:

    reward.terms:
      - {reward: {name: scorer, cmd: "uv run --script cookbook/rewards/scorers/finches.py --mode homotypic"},
         shaping: {name: quadratic, target: -6.0, width: 1.0}, label: eps, weight: 1.0}

Epsilon is unbounded, so pair it with entropy and length terms.

    --forcefield mpipi | calvados     which coarse-grained force field epsilon is derived from.
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
