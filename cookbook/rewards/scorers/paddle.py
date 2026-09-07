# /// script
# requires-python = ">=3.9,<3.12"
# dependencies = ["tensorflow==2.11.*", "numpy<2", "pandas"]
# ///
"""Score PADDLE-noSS activation-domain strength (https://github.com/asanborn/PADDLE).

Return the maximum Z-score over 53-residue windows; pad shorter sequences on both sides.
IDIOM_PADDLE_DIR sets the checkout cache (default ~/.cache/idiom/paddle).
PADDLE_STRIDE sets the window stride (default 5).
Run with uv run --script; model files are cloned on first use.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/asanborn/PADDLE.git"
WINDOW = 53          # PADDLE-noSS scores a fixed 53-residue window
_STRIDE = int(os.environ.get("PADDLE_STRIDE", "5"))
_PAD = "G"           # neutral flank when a sequence is shorter than one window


def _paddle_dir() -> Path:
    """Return the PADDLE checkout, cloning it on first use.

    Returns:
        A directory holding paddle.py and models/.
    """
    d = Path(os.environ.get("IDIOM_PADDLE_DIR", Path.home() / ".cache/idiom/paddle")).expanduser()
    if (d / "paddle.py").exists():
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning PADDLE (36 MB of model files, Apache-2.0) into {d} ...", file=sys.stderr,
          flush=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(d)], check=True,
                   stdout=sys.stderr.fileno(), stderr=sys.stderr.fileno())
    return d


def _windows(seq: str) -> list[str]:
    """Tile a sequence into 53-residue windows, padding a short sequence to one window."""
    if len(seq) < WINDOW:
        short = WINDOW - len(seq)
        left = short // 2
        return [_PAD * left + seq + _PAD * (short - left)]
    starts = list(range(0, len(seq) - WINDOW + 1, _STRIDE))
    if starts[-1] != len(seq) - WINDOW:
        starts.append(len(seq) - WINDOW)  # always include the final window
    return [seq[s:s + WINDOW] for s in starts]


def build():
    """Clone PADDLE if needed, load the model, and return the max-Z window scorer."""
    d = _paddle_dir()
    sys.path.insert(0, str(d))
    os.chdir(d)  # paddle.load_models resolves models/ relative to the working directory
    import numpy as np
    import paddle as paddle_module

    model = paddle_module.PADDLE_noSS()

    def score_batch(sequences):
        """Return the strongest 53-residue window Z-score per sequence."""
        flat, owner = [], []
        for i, seq in enumerate(sequences):
            for w in _windows(seq):
                flat.append(w)
                owner.append(i)
        # PADDLE returns a bare float for a single window and an array otherwise; normalize, or a
        # one-window batch (the startup handshake, for one) would not be iterable.
        preds = np.atleast_1d(model.predict(flat))  # one batched forward over every window
        best: dict[int, float] = {}
        for i, z in zip(owner, preds):
            best[i] = max(best.get(i, float("-inf")), float(z))
        return [best.get(i, 0.0) for i in range(len(sequences))]

    return score_batch


def serve(build):
    """Serve newline-delimited JSON requests until stdin closes.

    Call build() once to obtain a batch scorer. Redirect library output to stderr,
    score empty sequences as 0, and report scoring exceptions as JSON errors.
    """
    # This file's own directory is sys.path[0]; drop it so a scorer named after the package it wraps
    # (sparrow.py importing sparrow) resolves to the installed package, not back to itself.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    # stdout is the protocol. A library that prints on import (TensorFlow, ProtGPS, STARLING) would
    # corrupt the first response, so keep the real stdout for responses and send chatter to stderr.
    out, sys.stdout = sys.stdout, sys.stderr
    score_batch = build()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            seqs = json.loads(line)["sequences"]
            keep = [(i, s) for i, s in enumerate(seqs) if s]  # empty completions score 0.0
            values = score_batch([s for _, s in keep]) if keep else []
            scores = [0.0] * len(seqs)
            for (i, _), v in zip(keep, values):
                scores[i] = float(v)
            payload = {"scores": scores}
        except Exception as e:
            payload = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(payload), file=out, flush=True)


if __name__ == "__main__":
    serve(build)
