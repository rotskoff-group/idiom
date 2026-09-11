# /// script
# requires-python = ">=3.10"
# dependencies = ["biopython>=1.83", "numpy<2"]
# ///
"""Template for an external scorer with its own dependencies; copy and adapt.

Copy this file and _scorer_protocol.py into the same directory. Edit the dependency
header and build() below. This example scores Biopython isoelectric point or molecular
weight; IDiom applies shaping and weights separately.

In the reward YAML, set reward.name to external_scorer and reward.cmd to
"uv run --script /path/to/my_scorer.py" plus any arguments. Install uv with
`python -m pip install uv` if needed. The command runs directly, without shell expansion.
See cookbook/scripts/training/grpo/custom_scorer.yaml for a complete configuration.

The scorer persists across batches. Set reward.cache_max to 0 for stochastic scores;
set reward.timeout to allow for model loading and prediction. Use reward.env to select
a scorer GPU when needed. Test from the IDiom environment before training:

    python -m idiom.train.grpo.reward.external --cmd "uv run --script /path/to/my_scorer.py"
"""

import argparse

from _scorer_protocol import serve  # Keep the helper beside this file; it handles JSON communication


def build():
    """Load dependencies once and return a batch scorer; serve() redirects library output to stderr."""
    # EDIT: expose the settings your scorer needs as command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="isoelectric_point",
                    choices=["isoelectric_point", "molecular_weight"])
    prop = ap.parse_args().property # Options come from the YAML cmd field

    # EDIT: import dependencies and load model weights here, once per process rather than per batch
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    def score_batch(sequences):
        """Return one property value per non-empty sequence; let serve() report errors."""
        # EDIT: replace this calculation; batch model predictions here if supported
        # Inputs are non-empty; return one finite numeric score per sequence in the same order
        out = []
        for seq in sequences:
            analysis = ProteinAnalysis(seq)
            out.append(analysis.isoelectric_point() if prop == "isoelectric_point"
                       else analysis.molecular_weight())
        return out # Raise on invalid inputs rather than returning NaN or dropping results

    return score_batch # Return the function itself; serve() calls it for each incoming batch


if __name__ == "__main__":
    # Keep this entry point: empty sequences score 0, and library output is redirected to stderr
    # Use stderr for logs so stdout remains reserved for JSON responses
    serve(build)
