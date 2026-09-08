"""CPU metapredict V3 disorder diagnostics for generated sequences."""

from __future__ import annotations

import numpy as np
import torch


def disorder_totals(sequences: list[str]) -> tuple[float, int, int]:
    """Return sum of per-sequence mean disorder, nonempty count, and total count.

    Empty completions are excluded from disorder prediction and counted separately. Preserve
    Torch's CPU RNG: metapredict lazily constructs its network on the first prediction.
    """
    nonempty = [seq for seq in sequences if seq]
    if not nonempty:
        return 0.0, 0, len(sequences)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        import metapredict

        predictions = metapredict.predict_disorder_batch(
            nonempty, version="V3", device="cpu", normalized=True, round_values=False,
            return_numpy=True, show_progress_bar=False,
        )
    if len(predictions) != len(nonempty):
        raise ValueError("metapredict returned the wrong number of predictions")
    total = 0.0
    for expected, (sequence, scores) in zip(nonempty, predictions):
        scores = np.asarray(scores)
        if sequence != expected or scores.shape != (len(expected),):
            raise ValueError("metapredict returned misaligned residue scores")
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("metapredict returned invalid disorder scores")
        total += float(scores.mean())
    return total, len(nonempty), len(sequences)
