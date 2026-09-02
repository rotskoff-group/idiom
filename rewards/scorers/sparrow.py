"""Score IDRs with sparrow (https://github.com/idptools/sparrow) as an external GRPO reward.

The worked example of an external reward: it runs in its own environment and imports nothing from
IDiom. The config's cmd lets uv build sparrow on demand (no install step); see rewards/scorers/
README.md for cache and pinning guidance. In configs/grpo.yaml:

    reward.external:
      - {enabled: true, weight: 0.5, target: 25, width: 3,
         cmd: "uv run --isolated --no-project --with 'sparrow @ git+https://github.com/idptools/sparrow.git' python rewards/scorers/sparrow.py --property radius_of_gyration"}

It returns the raw property value (the term's target/width band it into a reward). --property is any
ALBATROSS predictor (radius_of_gyration, end_to_end_distance, asphericity, scaling_exponent,
prefactor) or a sequence parameter (FCR, NCPR, kappa, SCD, complexity). Note kappa returns -1.0 for a
sequence with no charged residues (a sentinel a policy can reach by removing all charge), so target
it only alongside an FCR constraint.
"""

import argparse
import json
import os
import sys

# This file is named sparrow.py, and Python puts a script's own directory on sys.path[0]; drop it so
# "import sparrow" resolves to the installed package rather than back to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

from sparrow import Protein  # noqa: E402


def value(sequence, prop):
    """Return the requested sparrow property for one sequence.

    ALBATROSS predictions are methods on Protein.predictor, while sequence parameters such as FCR
    and kappa are attributes on Protein itself. Checking the predictor first lets one flag select
    either kind.

    Args:
        sequence (str): An IDR residue string.
        prop (str): The property name.

    Returns:
        float: The property value.
    """
    protein = Protein(sequence)
    predictor = getattr(protein.predictor, prop, None)
    if callable(predictor):
        return float(predictor())
    return float(getattr(protein, prop))


def main():
    """Read batches from stdin and write scores to stdout until the parent closes the pipe."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="radius_of_gyration")
    prop = ap.parse_args().property

    try:  # name a bad property up front rather than once per batch
        value("MKVGSDEQ", prop)
    except AttributeError:
        print(f"unknown --property {prop!r}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"sparrow preflight failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            seqs = json.loads(line)["sequences"]
            response = {"scores": [value(s, prop) if s else 0.0 for s in seqs]}
        except Exception as e:
            response = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
