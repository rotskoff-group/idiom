"""Peak-aligned sequence windows and numerical logo data; rendering belongs to callers."""

from dataclasses import dataclass

import numpy as np

from idiom.data.tokenizer import RESIDUES as AMINO_ACIDS
from idiom.sae.features.feature_dataset import FeatureDataset


@dataclass
class FeatureWindow:
    """One sequence peak; '-' and NaN mark offsets without encoded residues."""

    sequence_id: int
    peak_position: int
    residues: str
    activations: np.ndarray


def feature_windows(
    dataset: FeatureDataset, feature_id: int, *, n: int = 60, half_width: int = 7
) -> list[FeatureWindow]:
    """Select one peak per firing sequence, ranked by peak then sequence ID.

    Offsets stay aligned to the peak. Only encoded residue positions contribute;
    sequence ends, FIM markers, and excluded regions remain missing. Short sequences
    are retained. Positions use the dataset's FIM coordinate system.
    """
    if not isinstance(half_width, int) or half_width < 0:
        raise ValueError("half_width must be a nonnegative integer")
    ids, _ = dataset.top_sequences(feature_id, n=n)
    offsets = np.arange(-half_width, half_width + 1)
    windows = []
    for seq_id in ids:
        positions, values = dataset.trace(int(seq_id), feature_id)
        peak = int(positions[values.argmax()])
        by_position = dict(zip(positions, values))
        sequence = dataset.sequence(int(seq_id))
        residues, activations = [], []
        for pos in peak + offsets:
            present = pos in by_position and sequence[pos] in AMINO_ACIDS
            residues.append(sequence[pos] if present else "-")
            activations.append(by_position[pos] if present else np.nan)
        windows.append(FeatureWindow(int(seq_id), peak, "".join(residues), np.array(activations)))
    return windows


def logo_data(dataset: FeatureDataset, feature_id: int, *, n: int = 60, half_width: int = 7) -> dict:
    """Return counts, information bits, and mean activations at fixed peak offsets.

    Counts have columns in AMINO_ACIDS order. Probabilities use available residues at
    each offset, without pseudocounts. Information uses a uniform amino-acid background.
    Mean activation divides by all selected windows (missing positions contribute zero).
    All arrays remain full width, including empty columns; no firing yields all zeros.
    """
    windows = feature_windows(dataset, feature_id, n=n, half_width=half_width)
    counts = np.zeros((2 * half_width + 1, len(AMINO_ACIDS)), dtype=np.int64)
    profile = np.zeros(len(counts))
    for window in windows:
        for i, aa in enumerate(window.residues):
            if aa != "-":
                counts[i, AMINO_ACIDS.index(aa)] += 1
        profile += np.nan_to_num(window.activations)
    total = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total > 0)
    logp = np.zeros_like(probabilities)
    np.log2(probabilities, out=logp, where=probabilities > 0)
    information = probabilities * (np.log2(len(AMINO_ACIDS)) + (probabilities * logp).sum(1, keepdims=True))
    return dict(
        windows=windows,
        offsets=np.arange(-half_width, half_width + 1),
        counts=counts,
        information=information,
        mean_activation=profile / max(len(windows), 1),
    )
