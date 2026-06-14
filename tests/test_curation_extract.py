"""P1 curation test (CPU-only): pLDDT -> IDR segmentation. No pipeline run."""

import numpy as np

from idiom.data.curation.extract import extract_idrs


def test_single_central_idr():
    # folded (40) | disordered (80) | folded (40): one clear IDR in the middle.
    plddt = np.concatenate([np.full(40, 90.0), np.full(80, 30.0), np.full(40, 90.0)])
    spans = extract_idrs(plddt)
    assert len(spans) == 1
    start, end = spans[0]
    assert end > start  # half-open, non-empty
    assert 30 <= (end - start) <= 100  # within band, ~the 80-residue dip widened by window blur
    assert 20 <= start and end <= 140  # localized to the central disordered stretch


def test_all_folded_returns_none():
    assert extract_idrs(np.full(100, 95.0)) == []


def test_short_disorder_dropped():
    # a 15-residue dip is below min_idr_length (30) -> not an IDR.
    plddt = np.concatenate([np.full(50, 90.0), np.full(15, 30.0), np.full(50, 90.0)])
    assert extract_idrs(plddt) == []
