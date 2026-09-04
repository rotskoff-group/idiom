# /// script
# requires-python = ">=3.10"
# dependencies = ["biopython>=1.83", "numpy<2"]
# ///
"""Template for a reward model in its own environment. Copy this file and edit it.

**Use this only when the scorer cannot be imported into IDiom's environment.** If it can, a term
naming an in-process function is simpler and much faster -- see my_rewards.py. A subprocess to call
something you could have imported is pure overhead. The reason this mechanism exists is that some
predictors pin dependencies IDiom cannot hold: ProtGPS wants python 3.8 with torch 2.0, PADDLE wants
TensorFlow. This script pins `numpy<2` to stand in for that -- a real and common source of conflict.

A scorer is a standalone program. It imports nothing from IDiom, so it can be a uv script (as here),
a conda environment, or `docker run -i`. Its dependencies live in the PEP 723 header above, so `uv
run --script` builds and caches the environment on first use and there is no install step.

    reward.terms:
      - {cmd: "uv run --script /path/to/my_scorer.py --property isoelectric_point",
         label: pI, weight: 1.0, shaping: {type: gaussian, target: 4.5, width: 0.3}}

The protocol is newline-delimited JSON, one exchange per GRPO step:

    ->  {"sequences": ["ACDEF...", "GHIKL..."]}
    <-  {"scores": [4.21, 9.87]}          # or {"error": "..."}

One finite score per sequence, in the order they were sent (a count mismatch is rejected), flushed
after each response. Load the model once at import, not per batch.

Check it before it ever takes a GPU -- this builds the environment, runs the handshake, and prints
what the shaping does to the raw value:

    uv run python -m idiom.train.grpo.reward.external \
        --cmd "uv run --script cookbook/rewards/my_scorer.py --property isoelectric_point" \
        --shaping gaussian --target 4.5 --width 0.3

What this one computes: isoelectric point and molecular weight, from Biopython. pI is a real handle
on IDR behaviour -- most nuclear IDRs are acidic, and pI tracks the charge balance that drives
complex coacervation -- and it is not one line of python, which is why it makes a better example
than something you would have written in-process.
"""

import argparse
import json
import os
import sys

# TRAP 1: this file's own directory is sys.path[0], so a script named after the package it wraps
# shadows it ("my_scorer.py" is safe; "sparrow.py" importing sparrow is not). Dropping it is free
# insurance, and you will need it as soon as you rename this file after your predictor.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

# TRAP 2: stdout is the protocol. A library that prints on import (TensorFlow, ProtGPS, STARLING)
# would corrupt the very first response and the run would die at the handshake with "scorer wrote a
# non-JSON line". Claim the real stdout now and send everything else to stderr, which the trainer
# forwards to its own log with this term's label.
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr

from Bio.SeqUtils.ProtParam import ProteinAnalysis  # noqa: E402 - after the sys.path/stdout fixes

# TRAP 3: load the model once, here, not inside score(). The child process persists across the whole
# run, so import-time cost is paid once; per-batch cost is paid 3000 times.


def score(sequence: str, prop: str) -> float:
    """Return the requested property for one sequence.

    Args:
        sequence (str): An IDR residue string of canonical amino acids.
        prop (str): "isoelectric_point" or "molecular_weight".

    Returns:
        float: The property value.

    Raises:
        ValueError: If prop is not one of the two supported properties.
    """
    analysis = ProteinAnalysis(sequence)
    if prop == "isoelectric_point":
        return float(analysis.isoelectric_point())
    if prop == "molecular_weight":
        return float(analysis.molecular_weight())
    raise ValueError(f"unknown --property {prop!r}")


def main() -> None:
    """Read batches from stdin and write scores to stdout until the parent closes the pipe."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="isoelectric_point",
                    choices=["isoelectric_point", "molecular_weight"])
    prop = ap.parse_args().property

    # Fail on a bad configuration once, at startup, rather than once per batch.
    score("MKVGSDEQ", prop)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            seqs = json.loads(line)["sequences"]
            # An empty completion still needs a score: return a neutral value rather than raising,
            # or one bad rollout takes down the run.
            response = {"scores": [score(s, prop) if s else 0.0 for s in seqs]}
        except Exception as e:
            response = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(response), file=_PROTOCOL_STDOUT, flush=True)


if __name__ == "__main__":
    main()
