"""Shared artifact resolution and length-filtered sampling."""

from __future__ import annotations

import warnings
from pathlib import Path

from huggingface_hub import snapshot_download


def _resolve(name_or_path: str | Path) -> Path:
    """Return a local directory or download a Hub snapshot."""
    p = Path(name_or_path)
    if p.exists():
        return p
    return Path(snapshot_download(str(name_or_path)))



def _oversample(batch_fn, n: int, *, length_range: tuple[int, int] | None = None,
                max_oversample: int = 20, seed: int | None = None) -> list[str]:
    """Draw up to n sequences within an inclusive length range.

    Without a range, draw once. Warn and return fewer sequences if the draw cap is reached.

    Args:
        batch_fn (Callable): Draws a batch of sequences given (k, seed).
        n: Number of sequences to return.
        length_range: Inclusive (lo, hi) length filter, or None.
        max_oversample: Cap on total draws, as a multiple of n.
        seed: Base seed, incremented once per re-draw.

    Returns:
        Up to n sequences, each within the length range if one was given.
    """
    if length_range is None:
        return batch_fn(n, seed)
    lo, hi = length_range
    kept: list[str] = []
    drawn, rounds, cap = 0, 0, n * max(1, max_oversample)
    while len(kept) < n and drawn < cap:
        s = None if seed is None else seed + rounds
        kept.extend(x for x in batch_fn(n, s) if x and lo <= len(x) <= hi)
        drawn += n
        rounds += 1
    if len(kept) < n:
        warnings.warn(f"generate: only {len(kept)}/{n} sequences fell in length {length_range} "
                      f"after {drawn} draws (max_oversample={max_oversample}); returning those.")
    return kept[:n]
