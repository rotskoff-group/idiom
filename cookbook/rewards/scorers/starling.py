# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = ["idptools-starling", "torch==2.4.1"]
# ///
"""Score STARLING ensemble dimensions (https://github.com/idptools/starling).

--property selects radius_of_gyration or end_to_end_distance; --conformations sets
ensemble size (default 20). STARLING_DEVICE overrides CUDA/CPU auto-selection.
Run with uv run --script. Ensemble generation is expensive; allow a generous scorer timeout.
The pinned torch build must match the CUDA driver.
"""

import argparse
import os

from _protocol import serve


def build():
    """Parse arguments and return the STARLING ensemble-dimension scorer."""
    ap = argparse.ArgumentParser(description="STARLING ensemble dimensions as an IDiom reward.")
    ap.add_argument("--property", default="radius_of_gyration",
                    choices=("radius_of_gyration", "end_to_end_distance"))
    ap.add_argument("--conformations", type=int, default=20)
    args = ap.parse_args()

    import torch
    from starling import generate

    device = os.environ.get("STARLING_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def score_batch(sequences):
        """Return the ensemble-average property per sequence."""
        keyed = {f"s{i}": s for i, s in enumerate(sequences)}
        ensembles = generate(keyed, conformations=args.conformations, device=device,
                             show_progress_bar=False)
        out = [0.0] * len(sequences)
        for key, ensemble in ensembles.items():
            out[int(key[1:])] = float(getattr(ensemble, args.property)(return_mean=True))
        return out

    return score_batch


if __name__ == "__main__":
    serve(build)
