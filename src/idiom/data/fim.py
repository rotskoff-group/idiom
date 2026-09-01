"""Fill-in-the-middle (FIM) formatting for IDiom (v2), applied on the fly.

A record is ``(full_seq, idr_start, idr_end)`` with **0-indexed, half-open** IDR coordinates
(``idr = full_seq[idr_start : idr_end]``, Python-slice native). The dataset assembles one of two FIM
strings per sample (the prompted/unprompted choice is a runtime augmentation, replacing the old
stored ×2):

- :func:`fim_prompted`    ``1{prefix}3{suffix}2{IDR}``  — an IDR conditioned on its flanking context
- :func:`fim_unprompted`  ``132{IDR}``                  — an IDR generated de novo, with no flanks

This matches the curation builder (``make_AFDB_FIM``) exactly. :func:`residue_source_positions`
gives, for each residue of a FIM string (markers dropped, in FIM order), its index back in
``full_seq`` — so extracted activations align 1:1 to the residues they came from.
"""

from __future__ import annotations

# Sentinels: 1 opens the prefix, 3 opens the suffix, 2 opens the middle (the IDR).
PREFIX, MIDDLE, SUFFIX = "1", "2", "3"

# Prompting-mode vocabulary. The two modes below name whether the generated IDR is conditioned on
# flanking context ("prompted") or produced de novo ("unprompted"). ``_MODE_ALIAS`` normalizes the
# older vocabulary (``idr``/``idp`` and ``context``/``denovo``) so old configs / call sites keep
# working. NB: this is a different axis from the biological IDR object (idr_start/idr_end, _IDR_x-y)
# and from the SAE ``region`` selector (all/idr/non_idr), neither of which is renamed.
PROMPTED, UNPROMPTED = "prompted", "unprompted"
_MODE_ALIAS = {"idr": PROMPTED, "context": PROMPTED, "idp": UNPROMPTED, "denovo": UNPROMPTED}


def normalize_mode(mode: str) -> str:
    """Map a prompting-mode string to the canonical ``prompted`` / ``unprompted`` (accepts aliases)."""
    m = _MODE_ALIAS.get(mode, mode)
    if m not in (PROMPTED, UNPROMPTED):
        raise ValueError(f"mode must be 'prompted' or 'unprompted' (or a legacy alias), got {mode!r}")
    return m


def fim_prompted(seq: str, start: int, end: int) -> str:
    """``1{prefix}3{suffix}2{IDR}`` — the IDR moved to the end, conditioned on both flanks."""
    prefix, idr, suffix = seq[:start], seq[start:end], seq[end:]
    return f"{PREFIX}{prefix}{SUFFIX}{suffix}{MIDDLE}{idr}"


def fim_unprompted(seq: str, start: int, end: int) -> str:
    """``132{IDR}`` — empty prefix and suffix, i.e. de-novo (unprompted) IDR generation."""
    return f"{PREFIX}{SUFFIX}{MIDDLE}{seq[start:end]}"


# Deprecated aliases (old vocabulary). Kept so existing imports keep working; prefer the names above.
fim_idr = fim_prompted
fim_idp = fim_unprompted


def fim_prompt(seq: str = "", start: int = 0, end: int = 0) -> str:
    """The generation prompt ``1{prefix}3{suffix}2`` — the model generates the IDR after ``2``.

    With the default empty sequence this is ``"132"`` (unprompted / de-novo). Given a protein + IDR
    span it is the flanking context, so the model in-fills an IDR conditioned on the flanks.
    """
    return f"{PREFIX}{seq[:start]}{SUFFIX}{seq[end:]}{MIDDLE}"


def residue_source_positions(seq_len: int, start: int, end: int, variant: str = PROMPTED) -> list[int]:
    """Source index in ``full_seq`` for each residue of the FIM string, in FIM order.

    "In FIM order" = the order residues appear once the ``1/3/2`` markers are dropped. For
    ``prompted`` (context) that is ``prefix, suffix, IDR``; for ``unprompted`` (de novo) it is just
    the IDR. Pairing these indices with the residue-only activation rows aligns every vector to its
    residue (and the IDR residues are exactly those with ``start <= pos < end``). Accepts the legacy
    ``idr``/``idp`` variant names.
    """
    variant = normalize_mode(variant)
    if variant == PROMPTED:
        return [*range(0, start), *range(end, seq_len), *range(start, end)]  # prefix, suffix, IDR
    return list(range(start, end))  # unprompted: IDR only
