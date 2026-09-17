"""Validation of public generation inputs before model execution."""

from __future__ import annotations

import math
from numbers import Integral, Real


def integer_at_least(name, value, minimum):
    """Require an integer at or above a lower bound.

    Args:
        name: Option name used in error messages.
        value: Integer to check; booleans are rejected.
        minimum: Inclusive lower bound.

    Raises:
        ValueError: If value is not an integer or is below minimum.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_sampling(temperature=1.0, top_k=None, top_p=None):
    """Validate sampling options, returning None on success.

    Args:
        temperature: Finite nonnegative number; zero selects greedy decoding.
        top_k: Positive integer cutoff, or None.
        top_p: Finite probability in (0, 1], or None.

    Raises:
        ValueError: If an option is outside its valid range or is a boolean.
    """
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and >= 0")
    if top_k is not None:
        integer_at_least("top_k", top_k, 1)
    if top_p is not None and (
        isinstance(top_p, bool)
        or not isinstance(top_p, Real)
        or not math.isfinite(top_p)
        or not 0 < top_p <= 1
    ):
        raise ValueError("top_p must be finite and in (0, 1]")


def validate_generation(
    n,
    *,
    max_new_tokens=1000,
    temperature=1.0,
    top_k=None,
    top_p=None,
    batch_size=None,
    length_range=None,
    max_oversample=20,
    seed=None,
):
    """Validate generation options, returning None on success.

    Booleans are rejected for numeric options. A zero sequence count is valid.

    Args:
        n: Nonnegative integer sequence count.
        max_new_tokens: Positive integer token budget per sequence.
        temperature: Finite nonnegative sampling temperature.
        top_k: Positive integer cutoff, or None.
        top_p: Finite nucleus cutoff in (0, 1], or None.
        batch_size: Positive integer batch size, or None.
        length_range: Optional pair of positive integer residue bounds (lo, hi),
            inclusive, with lo <= hi and lo <= max_new_tokens.
        max_oversample: Positive integer multiplier limiting length-filter draws.
        seed: Integer in [0, 2**64), or None.

    Raises:
        ValueError: If an option has an invalid type or value.
    """
    integer_at_least("n", n, 0)
    integer_at_least("max_new_tokens", max_new_tokens, 1)
    integer_at_least("max_oversample", max_oversample, 1)
    if batch_size is not None:
        integer_at_least("batch_size", batch_size, 1)
    if seed is not None:
        integer_at_least("seed", seed, 0)
        if seed >= 2**64:
            raise ValueError("seed must be less than 2**64")
    validate_sampling(temperature, top_k, top_p)
    if length_range is not None:
        if not isinstance(length_range, (tuple, list)) or len(length_range) != 2:
            raise ValueError("length_range must contain two integers (min_length, max_length)")
        lo, hi = length_range
        integer_at_least("length_range minimum", lo, 1)
        integer_at_least("length_range maximum", hi, 1)
        if lo > hi:
            raise ValueError("length_range minimum must be <= maximum")
        if lo > max_new_tokens:
            raise ValueError("length_range minimum exceeds max_new_tokens")
