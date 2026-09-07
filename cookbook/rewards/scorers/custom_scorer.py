# /// script
# requires-python = ">=3.10"
# dependencies = ["biopython>=1.83", "numpy<2"]
# ///
"""Template for a scorer with isolated dependencies; copy and adapt.

Run with uv run --script to install the PEP 723 dependencies. This example scores
Biopython isoelectric point or molecular weight, selected by --property.

Implement build() to load the model and return a batch scorer. Use the adjacent _scorer_protocol.py helper:
stdin accepts {"sequences": [...]}; stdout returns {"scores": [...]} or {"error": "..."},
one JSON object per line. Scores must be finite and match input order.
See cookbook/rewards/README.md for configuration and command checks.
"""

import argparse

from _scorer_protocol import serve


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


if __name__ == "__main__":
    serve(build)
