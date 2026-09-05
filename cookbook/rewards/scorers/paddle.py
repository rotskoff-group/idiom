# /// script
# requires-python = ">=3.9,<3.12"
# dependencies = ["tensorflow==2.11.*", "numpy<2", "pandas"]
# ///
"""Score IDRs with PADDLE (https://github.com/asanborn/PADDLE) as an external GRPO reward.

PADDLE is a convolutional network that predicts acidic transcriptional activation domains from
sequence (Sanborn et al., eLife 2021). This scorer uses PADDLE-noSS, the variant that runs from
sequence alone with no PSIPRED/IUPred structure input.

PADDLE scores a fixed 53-residue window, so an IDR is tiled into windows and the reward is the
strongest window's Z-score. A sequence shorter than 53 residues is padded on both flanks with the
background residue.

    reward.terms:
      - {reward: {name: scorer, cmd: "uv run --script cookbook/rewards/scorers/paddle.py",
                  timeout: 600}, label: paddle, weight: 1.0}

The raw reward is a Z-score against PADDLE's background; strong natural activation domains sit well
above 5. Leave it unshaped to maximize activation strength, or give it a quadratic target for a
particular strength.

The model files (36 MB, Apache-2.0) are cloned once from GitHub into IDIOM_PADDLE_DIR.

Environment variables:
    IDIOM_PADDLE_DIR   where the PADDLE checkout lives (default ~/.cache/idiom/paddle).
    PADDLE_STRIDE      window stride in residues (default 5; 1 is exhaustive and slower).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# This file is named paddle.py; drop its own directory from sys.path so the cloned PADDLE module is
# what "import paddle" finds, and keep stdout for the protocol since TensorFlow prints on import.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr

REPO_URL = "https://github.com/asanborn/PADDLE.git"
WINDOW = 53          # PADDLE-noSS scores a fixed 53-residue window
_STRIDE = int(os.environ.get("PADDLE_STRIDE", "5"))
_PAD = "G"           # neutral flank when a sequence is shorter than one window


def _paddle_dir() -> Path:
    """Return the PADDLE checkout, cloning it on first use.

    Returns:
        Path: A directory holding paddle.py and models/.
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


def main():
    """Read batches from stdin and write max-Z activation scores to stdout until the pipe closes."""
    d = _paddle_dir()
    sys.path.insert(0, str(d))
    os.chdir(d)  # paddle.load_models resolves models/ relative to the working directory
    import numpy as np
    import paddle as paddle_module

    model = paddle_module.PADDLE_noSS()

    def score_batch(sequences):
        """Return the strongest 53-residue window Z-score per sequence, 0.0 for empty strings."""
        scores = [0.0] * len(sequences)
        flat, owner = [], []
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            for w in _windows(seq):
                flat.append(w)
                owner.append(i)
        if not flat:
            return scores
        # PADDLE returns a bare float for a single window and an array otherwise; normalize, or a
        # one-sequence batch (the startup handshake, for one) would not be iterable.
        preds = np.atleast_1d(model.predict(flat))  # one batched forward over every window
        best: dict[int, float] = {}
        for i, z in zip(owner, preds):
            best[i] = max(best.get(i, float("-inf")), float(z))
        for i, z in best.items():
            scores[i] = z
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
