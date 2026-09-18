# /// script
# requires-python = ">=3.10"
# dependencies = ["finches @ git+https://github.com/idptools/finches.git"]
# ///
"""Score FINCHES interaction epsilon (https://github.com/idptools/finches).

Use --mode homotypic for self-interaction or --mode heterotypic --partner SEQUENCE.
Choose --forcefield mpipi or calvados. Negative epsilon indicates attraction.
Run with uv run --script; see cookbook/rewards/README.md for reward configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finches import CALVADOS_frontend, Mpipi_frontend

import argparse

from _scorer_protocol import serve


def build_frontend(forcefield) -> CALVADOS_frontend | Mpipi_frontend:
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


def build() -> Callable[[list[str]], list[float]]:
    """Parse arguments and return the epsilon scorer for the chosen mode and force field."""
    ap = argparse.ArgumentParser(description="FINCHES epsilon as an IDiom external reward.")
    ap.add_argument(
        "--mode",
        default="homotypic",
        choices=("homotypic", "heterotypic"),
        help="self-interaction, or interaction with --partner",
    )
    ap.add_argument("--partner", default=None, help="partner residue string, required for --mode heterotypic")
    ap.add_argument(
        "--forcefield",
        default="mpipi",
        choices=("mpipi", "calvados"),
        help="coarse-grained force field epsilon is derived from",
    )
    args = ap.parse_args()
    if args.mode == "heterotypic" and not args.partner:
        raise SystemExit("--mode heterotypic needs --partner <sequence>")

    frontend = build_frontend(args.forcefield)
    partner = args.partner if args.mode == "heterotypic" else None

    def score_batch(sequences) -> list[float]:
        """Return epsilon(seq, partner-or-self) for each sequence."""
        return [float(frontend.epsilon(s, partner or s)) for s in sequences]

    return score_batch


if __name__ == "__main__":
    serve(build)
