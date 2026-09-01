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

# Prompting-mode vocabulary. The two modes below name whether the generated IDR is conditioned on
# flanking context ("prompted") or produced de novo ("unprompted"). _MODE_ALIAS normalizes the
# older vocabulary ("idr"/"idp" and "context"/"denovo") so old configs / call sites keep
# working. NB: this is a different axis from the biological IDR object (idr_start/idr_end, _IDR_x-y)
# and from the SAE region selector (all/idr/non_idr), neither of which is renamed.
PROMPTED, UNPROMPTED = "prompted", "unprompted"
_MODE_ALIAS = {"idr": PROMPTED, "context": PROMPTED, "idp": UNPROMPTED, "denovo": UNPROMPTED}


def normalize_mode(mode: str) -> str:
    """Map a prompting-mode string to the canonical prompted / unprompted.

    Accepts the legacy aliases idr/context (-> prompted) and idp/denovo (-> unprompted).

    Args:
        mode (str): A prompting-mode string, canonical or legacy.

    Returns:
        str: Either "prompted" or "unprompted".

    Raises:
        ValueError: If mode is not a recognized mode or alias.
    """
    m = _MODE_ALIAS.get(mode, mode)
    if m not in (PROMPTED, UNPROMPTED):
        raise ValueError(f"mode must be 'prompted' or 'unprompted' (or a legacy alias), got {mode!r}")
    return m


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

    Prefix and suffix are empty, i.e. de-novo generation with no flanking context.

    Args:
        seq (str): The full protein sequence.
        start (int): IDR start index (0-based, inclusive).
        end (int): IDR end index (0-based, exclusive).

    Returns:
        str: The FIM-formatted string.
    """
    return f"{PREFIX}{SUFFIX}{MIDDLE}{seq[start:end]}"


# Deprecated aliases (old vocabulary). Kept so existing imports keep working; prefer the names above.
fim_idr = fim_prompted
fim_idp = fim_unprompted


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
        variant (str): "prompted" or "unprompted" (legacy "idr"/"idp" accepted).

    Returns:
        list[int]: The source position of each residue, in FIM order.
    """
    variant = normalize_mode(variant)
    if variant == PROMPTED:
        return [*range(0, start), *range(end, seq_len), *range(start, end)]  # prefix, suffix, IDR
    return list(range(start, end))  # unprompted: IDR only
