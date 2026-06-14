"""Fill-in-the-middle (FIM) formatting for IDiom (v2), applied on the fly.

A record is ``(full_seq, idr_start, idr_end)`` with **0-indexed, half-open** IDR coordinates
(``idr = full_seq[idr_start : idr_end]``, Python-slice native). The dataset assembles one of two FIM strings
per sample (the full/132 choice is a runtime augmentation, replacing the old stored ×2):

- :func:`fim_full`  ``1{prefix}3{suffix}2{IDR}``  — context-prompted
- :func:`fim_132`   ``132{IDR}``                  — de-novo (no flanking context)

This matches the curation builder (``make_AFDB_FIM``) exactly. :func:`residue_source_positions`
gives, for each residue of a FIM string (markers dropped, in FIM order), its index back in
``full_seq`` — so extracted activations align 1:1 to the residues they came from.
"""

from __future__ import annotations

# Sentinels: 1 opens the prefix, 3 opens the suffix, 2 opens the middle (the IDR).
PREFIX, MIDDLE, SUFFIX = "1", "2", "3"


def fim_full(seq: str, start: int, end: int) -> str:
    """``1{prefix}3{suffix}2{IDR}`` — the IDR moved to the end, conditioned on both flanks."""
    prefix, idr, suffix = seq[:start], seq[start:end], seq[end:]
    return f"{PREFIX}{prefix}{SUFFIX}{suffix}{MIDDLE}{idr}"


def fim_132(seq: str, start: int, end: int) -> str:
    """``132{IDR}`` — empty prefix and suffix, i.e. de-novo IDR/IDP generation."""
    return f"{PREFIX}{SUFFIX}{MIDDLE}{seq[start:end]}"


def fim_prompt(seq: str = "", start: int = 0, end: int = 0) -> str:
    """The generation prompt ``1{prefix}3{suffix}2`` — the model generates the IDR after ``2``.

    With the default empty sequence this is ``"132"`` (de-novo IDP). Given a protein + IDR span
    it is the flanking context, so the model in-fills an IDR conditioned on the flanks.
    """
    return f"{PREFIX}{seq[:start]}{SUFFIX}{seq[end:]}{MIDDLE}"


def residue_source_positions(seq_len: int, start: int, end: int, variant: str = "full") -> list[int]:
    """Source index in ``full_seq`` for each residue of the FIM string, in FIM order.

    "In FIM order" = the order residues appear once the ``1/3/2`` markers are dropped. For
    ``full`` that is ``prefix, suffix, IDR``; for ``132`` it is just the IDR. Pairing these
    indices with the residue-only activation rows aligns every vector to its residue (and the
    IDR residues are exactly those with ``start <= pos < end``).
    """
    if variant == "full":
        return [*range(0, start), *range(end, seq_len), *range(start, end)]  # prefix, suffix, IDR
    if variant == "132":
        return list(range(start, end))  # IDR only
    raise ValueError(f"variant must be 'full' or '132', got {variant!r}")
