# /// script
# requires-python = ">=3.10"
# dependencies = ["sparrow @ git+https://github.com/idptools/sparrow.git"]
# ///
"""Score sparrow sequence properties (https://github.com/idptools/sparrow).

--property selects an ALBATROSS prediction or a Protein attribute such as FCR or NCPR.
Kappa returns -1 for sequences without charged residues.
Run with uv run --script; see cookbook/rewards/README.md for reward configuration.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from _scorer_protocol import serve


def build() -> Callable[[list[str]], list[float]]:
    """Parse arguments and return the sparrow property scorer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="radius_of_gyration")
    prop = ap.parse_args().property
    from sparrow import Protein

    def score_batch(sequences) -> list[float]:
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


if __name__ == "__main__":
    serve(build)
