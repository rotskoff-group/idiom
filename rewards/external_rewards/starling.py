# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = ["idptools-starling", "torch==2.4.1"]
# ///
"""Score IDRs with STARLING (https://github.com/idptools/starling) as an external GRPO reward.

STARLING is a diffusion model that generates coarse-grained conformational ensembles for an IDR,
so the reward is an ensemble average rather than a regression: the same dimensions sparrow's
ALBATROSS predictors estimate directly, but sampled from a generated ensemble, plus anything else
an ensemble exposes.

    --property radius_of_gyration | end_to_end_distance
    --conformations N             ensemble size per sequence (default 20)

    reward.terms:
      - {cmd: "uv run --script rewards/external_rewards/starling.py --property radius_of_gyration",
         label: rg_ens, weight: 1.0, shaping: {type: quadratic, target: 25, width: 0.2},
         timeout: 900}

Cost: about 9 s per 32 sequences at 20 conformations on an H100, which roughly doubles GRPO step
time -- worth it for an ensemble-level objective, but give the term a generous timeout. On CPU it
is about 6 s per sequence, too slow to train against.

The torch pin matters: the wheel idptools-starling resolves by default can be newer than the host
CUDA driver, which fails at model load with "The NVIDIA driver on your system is too old". Pin the
torch build that matches your driver, or run with CUDA_VISIBLE_DEVICES="" to fall back to CPU.

Environment variables:
    STARLING_DEVICE   torch device (default cuda when visible, else cpu).
"""

import argparse
import json
import os
import sys

# This file is named starling.py, and Python puts a script's own directory on sys.path[0]; drop it
# so "import starling" resolves to the installed package rather than back to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

# STARLING prints to stdout ("Using DDIM sampler"), and stdout is the protocol: any stray line
# there is read as a malformed response. Keep the real stdout for responses and send everything
# else to stderr, which the parent forwards to its log.
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def main():
    """Read batches from stdin and write ensemble-average scores to stdout until the pipe closes."""
    ap = argparse.ArgumentParser(description="STARLING ensemble dimensions as an IDiom reward.")
    ap.add_argument("--property", default="radius_of_gyration",
                    choices=("radius_of_gyration", "end_to_end_distance"))
    ap.add_argument("--conformations", type=int, default=20)
    args = ap.parse_args()

    import torch
    from starling import generate

    device = os.environ.get("STARLING_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    def score_batch(sequences):
        """Return the ensemble-average property per sequence, 0.0 for empty strings."""
        keep = {f"s{i}": s for i, s in enumerate(sequences) if s}
        scores = [0.0] * len(sequences)
        if not keep:
            return scores
        ensembles = generate(keep, conformations=args.conformations, device=device,
                             show_progress_bar=False)
        for key, ensemble in ensembles.items():
            value = getattr(ensemble, args.property)(return_mean=True)
            scores[int(key[1:])] = float(value)
        return scores

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = {"scores": score_batch(json.loads(line)["sequences"])}
        except Exception as e:
            response = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(response), file=_PROTOCOL_STDOUT, flush=True)


if __name__ == "__main__":
    main()
