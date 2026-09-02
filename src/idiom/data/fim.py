"""Fill-in-the-middle (FIM) formatting for IDiom, applied on the fly.

A record is (full_seq, idr_start, idr_end) with 0-indexed, half-open IDR coordinates
(idr = full_seq[idr_start:idr_end]). The dataset assembles one of two FIM strings per sample;
the prompted/unprompted choice is a per-sample augmentation:

- fim_prompted   -> 1{prefix}3{suffix}2{IDR}: an IDR conditioned on its flanking context.
- fim_unprompted -> 132{IDR}: an IDR generated de novo, with no flanks.

residue_source_positions gives, for each residue of a FIM string (markers dropped, in FIM order),
its index back into full_seq, so extracted activations align 1:1 with the residues they came from.
"""

from __future__ import annotations

# Sentinels: 1 opens the prefix, 3 opens the suffix, 2 opens the middle (the IDR).
PREFIX, MIDDLE, SUFFIX = "1", "2", "3"

# Prompting-mode vocabulary: whether the generated IDR is conditioned on flanking context
# ("prompted") or produced de novo ("unprompted"). NB: this is a different axis from the biological
# IDR object (idr_start/idr_end, the _IDR_x-y header span) and from the SAE region selector
# (all/idr/non_idr) — "idr" means something different in each, and none is derived from the others.
PROMPTED, UNPROMPTED = "prompted", "unprompted"


def normalize_mode(mode: str) -> str:
    """Validate a prompting-mode string and return it unchanged.

    Every prompting-mode value passes through here — the dataset's per-sample variant, the
    extractor's fim_mode, and the fim_mode persisted in an SAE release — so a typo or an
    unrecognized mode fails loudly at the call site instead of silently selecting the wrong FIM
    format (which does not raise anywhere downstream; it just produces off-distribution output).

    Args:
        mode (str): A prompting-mode string.

    Returns:
        str: Either "prompted" or "unprompted".

    Raises:
        ValueError: If mode is neither "prompted" nor "unprompted".
    """
    if mode not in (PROMPTED, UNPROMPTED):
        raise ValueError(f"mode must be 'prompted' or 'unprompted', got {mode!r}")
    return mode


def fim_prompted(seq: str, start: int, end: int) -> str:
    """Build the prompted FIM string 1{prefix}3{suffix}2{IDR}.

    The IDR is moved to the end, conditioned on both flanks.

    Args:
        seq (str): The full protein sequence.
        start (int): IDR start index (0-based, inclusive).
        end (int): IDR end index (0-based, exclusive).

    Returns:
        str: The FIM-formatted string.
    """
    prefix, idr, suffix = seq[:start], seq[start:end], seq[end:]
    return f"{PREFIX}{prefix}{SUFFIX}{suffix}{MIDDLE}{idr}"


def fim_unprompted(seq: str, start: int, end: int) -> str:
    """Build the unprompted FIM string 132{IDR}.

    Prefix and suffix are empty, i.e. de novo generation with no flanking context.

    Args:
        seq (str): The full protein sequence.
        start (int): IDR start index (0-based, inclusive).
        end (int): IDR end index (0-based, exclusive).

    Returns:
        str: The FIM-formatted string.
    """
    return f"{PREFIX}{SUFFIX}{MIDDLE}{seq[start:end]}"


def fim_prompt(seq: str = "", start: int = 0, end: int = 0) -> str:
    """Build the generation prompt 1{prefix}3{suffix}2 (the model generates the IDR after 2).

    With the default empty sequence this is "132" (unprompted / de novo). Given a protein and an
    IDR span it becomes the flanking context, so the model in-fills an IDR conditioned on the flanks.

    Args:
        seq (str): The full protein sequence (empty for unprompted generation).
        start (int): IDR start index (0-based, inclusive).
        end (int): IDR end index (0-based, exclusive).

    Returns:
        str: The prompt string to feed the model.
    """
    return f"{PREFIX}{seq[:start]}{SUFFIX}{seq[end:]}{MIDDLE}"


def residue_source_positions(seq_len: int, start: int, end: int, variant: str = PROMPTED) -> list[int]:
    """Return the source index in full_seq for each residue of the FIM string, in FIM order.

    "FIM order" is the order residues appear once the 1/3/2 markers are dropped: for prompted that
    is prefix, suffix, IDR; for unprompted it is just the IDR. Pairing these indices with the
    residue-only activation rows aligns every vector with its residue (the IDR residues are exactly
    those with start <= pos < end).

    Args:
        seq_len (int): Length of the full sequence.
        start (int): IDR start index (0-based, inclusive).
        end (int): IDR end index (0-based, exclusive).
        variant (str): "prompted" or "unprompted".

    Returns:
        list[int]: The source position of each residue, in FIM order.
    """
    variant = normalize_mode(variant)
    if variant == PROMPTED:
        return [*range(0, start), *range(end, seq_len), *range(start, end)]  # prefix, suffix, IDR
    return list(range(start, end))  # unprompted: IDR only
