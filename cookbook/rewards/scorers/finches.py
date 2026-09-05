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
      - {cmd: "uv run --script cookbook/rewards/scorers/finches.py --mode homotypic",
         label: eps, weight: 1.0, shaping: {type: quadratic, target: -6.0, width: 1.0}}

Epsilon is unbounded, so pair it with entropy and length terms.

    --forcefield mpipi | calvados     which coarse-grained force field epsilon is derived from.
"""

import argparse
import json
import os
import sys

# This file is named finches.py, and Python puts a script's own directory on sys.path[0]; drop it
# so "import finches" resolves to the installed package rather than back to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]


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


def main():
    """Read batches from stdin and write epsilon scores to stdout until the pipe closes."""
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

    def score(seq):
        """Return epsilon for one sequence, or 0.0 for an empty string."""
        if not seq:
            return 0.0
        other = args.partner if args.mode == "heterotypic" else seq
        return float(frontend.epsilon(seq, other))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = {"scores": [score(s) for s in json.loads(line)["sequences"]]}
        except Exception as e:
            response = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
