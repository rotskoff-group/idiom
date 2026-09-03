"""Cutting the residue windows a sequence logo is built from.

A feature's meaning is easiest to read as a logo over the windows where it fires hardest. Both
functions here work on the in-memory output of IDiomSAE.encode(pool="none") -- a [N_res, latents]
activation matrix plus its per-row index -- rather than on a feature dataset directory, because a
logo is drawn over one set of sequences small enough to hold at once.
"""

from __future__ import annotations

import numpy as np


def per_sequence_activations(feats, index) -> list[tuple[str, np.ndarray]]:
    """Group per-residue rows back into per-sequence residues and row indices.

    encode(pool="none") returns residue rows in order across the whole set; regrouping them lets a
    window be cut from a single sequence's residue string.

    Args:
        feats (np.ndarray): [N_res, num_latents] per-residue activations. Unused except to fix the
            row convention shared with index.
        index (list[dict]): Per-row metadata carrying accession, source_pos, and residue.

    Returns:
        list[tuple[str, np.ndarray]]: One (residue_string, row_indices) per accession, in the order
            the accessions first appear.
    """
    order: list[str] = []
    rows: dict[str, list[int]] = {}
    for i, row in enumerate(index):
        acc = row["accession"]
        if acc not in rows:
            rows[acc] = []
            order.append(acc)
        rows[acc].append(i)
    out = []
    for acc in order:
        idx = sorted(rows[acc], key=lambda i: index[i]["source_pos"])
        residues = "".join(index[i]["residue"] for i in idx)
        out.append((residues, np.array(idx)))
    return out


def top_windows(feature_id, feats, per_seq, *, n_windows=60, half_width=7) -> list[str]:
    """Return the top-activating fixed-width residue windows for one feature.

    One window per sequence, centred on that sequence's peak activation and clamped to fit, so the
    stack is equal-length and no sequence dominates it.

    Args:
        feature_id (int): The SAE latent to profile.
        feats (np.ndarray): [N_res, num_latents] activations.
        per_seq (list): Output of per_sequence_activations.
        n_windows (int): Maximum windows to return.
        half_width (int): Residues on each side of the peak; window length is 2*half_width+1.

    Returns:
        list[str]: Equal-length residue windows, most-active first. Sequences shorter than the
            window, and sequences where the feature never fires, are left out.
    """
    length = 2 * half_width + 1
    peaks = []
    for residues, idx in per_seq:
        if len(residues) < length:
            continue  # too short to cut a full window
        acts = feats[idx, feature_id]
        p = int(acts.argmax())
        peaks.append((float(acts[p]), residues, p))
    peaks.sort(key=lambda t: t[0], reverse=True)
    windows = []
    for act, residues, p in peaks[:n_windows]:
        if act <= 0:
            break  # feature never fires beyond here
        start = min(max(p - half_width, 0), len(residues) - length)  # clamp so the window fits
        windows.append(residues[start:start + length])
    return windows
