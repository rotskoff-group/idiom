"""Standard-library-only scorer protocol, compatible with Python 3.8 and later."""

import json
import math
import os
import sys


def serve(build):
    """Load a scorer once and serve validated JSON batches until stdin closes."""
    # Import this helper before removing the directory: sparrow.py must not shadow sparrow
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    out, sys.stdout = sys.stdout, sys.stderr
    try:
        score_batch = build()
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                seqs = request.get("sequences") if isinstance(request, dict) else None
                if not isinstance(seqs, list) or any(not isinstance(s, str) for s in seqs):
                    raise ValueError("sequences must be a list of strings")
                keep = [(i, s) for i, s in enumerate(seqs) if s]
                values = list(score_batch([s for _, s in keep])) if keep else []
                if len(values) != len(keep):
                    raise ValueError(f"scorer returned {len(values)} scores for {len(keep)} sequences")
                scores = [0.0] * len(seqs)
                for (i, _), value in zip(keep, values):
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"score {i} is not finite")
                    scores[i] = value
                payload = {"scores": scores}
            except Exception as e:
                payload = {"error": f"{type(e).__name__}: {e}"}
            print(json.dumps(payload, allow_nan=False), file=out, flush=True)
    finally:
        sys.stdout = out
