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

import os
import subprocess
import sys
from pathlib import Path

from _scorer_protocol import serve

REPO_URL = "https://github.com/asanborn/PADDLE.git"
WINDOW = 53  # PADDLE-noSS scores a fixed 53-residue window
_STRIDE = int(os.environ.get("PADDLE_STRIDE", "5"))
_PAD = "G"  # neutral flank when a sequence is shorter than one window


def _paddle_dir() -> Path:
    """Return the PADDLE checkout containing paddle.py and models/, cloning it if needed."""
    d = Path(os.environ.get("IDIOM_PADDLE_DIR", Path.home() / ".cache/idiom/paddle")).expanduser()
    if (d / "paddle.py").exists():
        return d
    d.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning PADDLE (36 MB of model files, Apache-2.0) into {d} ...", file=sys.stderr, flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(d)],
        check=True,
        stdout=sys.stderr.fileno(),
        stderr=sys.stderr.fileno(),
    )
    return d


def _windows(seq: str) -> list[str]:
    """Tile a sequence into 53-residue windows, padding a short sequence to one window."""
    if len(seq) < WINDOW:
        short = WINDOW - len(seq)
        left = short // 2
        return [_PAD * left + seq + _PAD * (short - left)]
    starts = list(range(0, len(seq) - WINDOW + 1, _STRIDE))
    if starts[-1] != len(seq) - WINDOW:
        starts.append(len(seq) - WINDOW)
    return [seq[s : s + WINDOW] for s in starts]


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
        # PADDLE returns a scalar for a single window
        preds = np.atleast_1d(model.predict(flat))
        best: dict[int, float] = {}
        for i, z in zip(owner, preds):
            best[i] = max(best.get(i, float("-inf")), float(z))
        return [best.get(i, 0.0) for i in range(len(sequences))]

    return score_batch


if __name__ == "__main__":
    serve(build)
