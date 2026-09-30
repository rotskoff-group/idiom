"""Fill-in-the-middle (FIM) string formatting.

A record is (full_seq, idr_start, idr_end) with 0-indexed, half-open IDR coordinates, so
idr = full_seq[idr_start:idr_end]. The FIM forms of a record are:

- fim_prompted: "1{prefix}3{suffix}2{IDR}", the IDR with its flanking context.
- fim_unprompted: "132{IDR}", the IDR with empty flanks.
- fim_prompt: "1{prefix}3{suffix}2", the generation prompt.
"""

from __future__ import annotations

PREFIX, MIDDLE, SUFFIX = "1", "2", "3"

# Prompting mode controls flanking context; SAE region controls residue selection
PROMPTED, UNPROMPTED = "prompted", "unprompted"


def normalize_mode(mode: str) -> str:
    """Return a valid prompting mode unchanged.

    Raises:
        ValueError: If mode is not "prompted" or "unprompted".
    """
    if mode not in (PROMPTED, UNPROMPTED):
        raise ValueError(f"mode must be 'prompted' or 'unprompted', got {mode!r}")
    return mode


def fim_prompted(seq: str, start: int, end: int) -> str:
    """Return "1{prefix}3{suffix}2{IDR}" using the 0-based half-open span seq[start:end]."""
    prefix, idr, suffix = seq[:start], seq[start:end], seq[end:]
    return f"{PREFIX}{prefix}{SUFFIX}{suffix}{MIDDLE}{idr}"


def fim_unprompted(seq: str, start: int, end: int) -> str:
    """Return "132{IDR}" using the 0-based half-open span seq[start:end]."""
    return f"{PREFIX}{SUFFIX}{MIDDLE}{seq[start:end]}"


def fim_prompt(seq: str = "", start: int = 0, end: int = 0) -> str:
    """Return "1{prefix}3{suffix}2", omitting the 0-based half-open span seq[start:end].

    Empty defaults produce the unprompted prefix "132".
    """
    return f"{PREFIX}{seq[:start]}{SUFFIX}{seq[end:]}{MIDDLE}"


def residue_source_positions(seq_len: int, start: int, end: int, variant: str = PROMPTED) -> list[int]:
    """Map FIM-ordered residues to source-sequence positions.

    FIM order is the order residues appear once the 1/3/2 markers are dropped: prefix, suffix, IDR
    for "prompted", and the IDR alone for "unprompted".

    Args:
        seq_len: Length of the full sequence.
        start: IDR start index (0-based, inclusive).
        end: IDR end index (0-based, exclusive).
        variant: "prompted" or "unprompted".

    Returns:
        The source position of each residue, in FIM order.

    Raises:
        ValueError: If variant is neither "prompted" nor "unprompted".
    """
    variant = normalize_mode(variant)
    if variant == PROMPTED:
        return [*range(0, start), *range(end, seq_len), *range(start, end)]
    return list(range(start, end))
