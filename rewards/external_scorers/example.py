"""Template for an external reward scorer. Copy this file and change score_batch.

Reads one JSON line per batch from stdin, writes one per batch to stdout, importing nothing from
IDiom:

    ->  {"sequences": ["ACDEF...", "GHIKL..."]}
    <-  {"scores": [0.12, 0.98]}

    uv run python -m idiom.train.grpo.reward.external_reward --cmd "python rewards/external_scorers/example.py"

Flush after every response, return one score per sequence in order, and send logging to stderr
(stdout is the protocol). Load the model once at import -- the process is reused for the whole run.
"""

import json
import sys


def score_batch(sequences):
    """Return one score per sequence. Replace this with your model.

    Args:
        sequences (list[str]): IDR residue strings.

    Returns:
        list[float]: One finite score per sequence, in the order received.
    """
    return [sum(s.count(a) for a in "FWY") / len(s) if s else 0.0 for s in sequences]


def main():
    """Read batches from stdin and write scores to stdout until the parent closes the pipe."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            scores = score_batch(json.loads(line)["sequences"])
            response = {"scores": [float(s) for s in scores]}
        except Exception as e:  # report failures in-band so the parent can raise a clear error
            response = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
