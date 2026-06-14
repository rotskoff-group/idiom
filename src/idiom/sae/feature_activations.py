"""Reductions over the feature-activation dataset produced by `build_feature_dataset`.

A directory (no h5) of top-k sparse SAE activations per residue, with ``(seq_idx, pos_idx)``
joins back to the raw FIM sequence strings::

    top_indices.npy  int32[N_res, k]   which latents fired at each residue
    top_values.npy   float32[N_res, k] their activations
    seq_idx.npy      int32[N_res]      index into strings
    pos_idx.npy      int32[N_res]      position within strings[s] (the FIM string)
    strings.json     list[str]         raw FIM `1{prefix}3{suffix}2{IDR}` per sequence
    meta.json        {k, num_latents, layer}

These helpers are the building blocks the visualizer / annotator / concept correlator use to
slice that artifact — no streaming, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatureDataset:
    """Reader over a feature-activation HDF5.

    Opens once, exposes the per-residue arrays plus a few common reductions.
    By default the arrays stay memory-mapped (slicing hits disk). Pass
    ``in_memory=True`` to pull them fully into RAM, in which case every reduction
    below operates on numpy arrays with no further disk I/O — slicing such as
    ``fd.top_indices[:]`` then becomes a cheap view instead of a disk read. This
    is what the Streamlit viewer wants, since it rereads the whole script (and
    so reslices these arrays) on every widget change. Use as a context manager
    or close explicitly.
    """

    def __init__(self, path: str | Path, in_memory: bool = False):
        self.path = Path(path)
        self.in_memory = in_memory
        # mmap when not in_memory (lazy, slicing hits disk); full load otherwise.
        mmap = None if in_memory else "r"
        self.top_indices = np.load(self.path / "top_indices.npy", mmap_mode=mmap)
        self.top_values = np.load(self.path / "top_values.npy", mmap_mode=mmap)
        self.seq_idx = np.load(self.path / "seq_idx.npy", mmap_mode=mmap)
        self.pos_idx = np.load(self.path / "pos_idx.npy", mmap_mode=mmap)
        self.strings = json.loads((self.path / "strings.json").read_text())
        meta = json.loads((self.path / "meta.json").read_text())
        self.k = int(meta["k"])
        self.num_latents = int(meta["num_latents"])
        self.layer = int(meta["layer"])

    def __enter__(self) -> FeatureDataset:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        # npy mmaps are released on GC; nothing to close explicitly.
        pass

    def sequence(self, local_seq_idx: int) -> str:
        return str(self.strings[int(local_seq_idx)])


def feature_activations_at_rows(
    fd: FeatureDataset, feature_id: int, rows: np.ndarray
) -> np.ndarray:
    """Return feature ``feature_id``'s activation at each row in ``rows`` (0 if not in top-k)."""
    rows = np.asarray(rows, dtype=np.int64)
    # h5py fancy-indexing requires strictly increasing indices, but callers may
    # pass rows in any order (e.g. sorted by position). Read in sorted order,
    # then restore the caller's original ordering before returning.
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    top_idx = np.asarray(fd.top_indices[sorted_rows])  # [len(rows), k]
    top_val = np.asarray(fd.top_values[sorted_rows])
    mask = top_idx == int(feature_id)
    acts_sorted = (top_val * mask).sum(axis=1)
    acts = np.empty_like(acts_sorted)
    acts[order] = acts_sorted
    return acts


def feature_trace_for_sequence(
    fd: FeatureDataset, local_seq_idx: int, feature_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct feature ``feature_id``'s per-residue activation across one sequence.

    Returns ``(positions, activations)`` sorted by position. Residues where the
    feature was not in the top-k contribute zero.
    """
    seq_idx_all = np.asarray(fd.seq_idx[:])
    rows = np.where(seq_idx_all == int(local_seq_idx))[0]
    if len(rows) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    pos = np.asarray(fd.pos_idx[rows], dtype=np.int64)
    order = np.argsort(pos)
    rows = rows[order]
    pos = pos[order]
    acts = feature_activations_at_rows(fd, feature_id, rows)
    return pos, acts


def per_sequence_peak(
    fd: FeatureDataset, feature_id: int, num_sequences: int | None = None
) -> np.ndarray:
    """Per-sequence peak activation of feature ``feature_id``.

    Returns ``peak[S]`` where ``S = num_sequences`` (default: full sequence
    table). Sequences with no firing residues get 0.
    """
    S = int(num_sequences) if num_sequences is not None else len(fd.strings)
    seq_idx_all = np.asarray(fd.seq_idx[:], dtype=np.int64)
    top_idx = np.asarray(fd.top_indices[:])
    top_val = np.asarray(fd.top_values[:])
    mask = top_idx == int(feature_id)
    row_acts = (top_val * mask).sum(axis=1)
    peak = np.zeros(S, dtype=np.float32)
    np.maximum.at(peak, seq_idx_all, row_acts)
    return peak


def per_sequence_fraction_firing(
    fd: FeatureDataset,
    feature_id: int,
    num_sequences: int | None = None,
    threshold: float = 0.0,
) -> np.ndarray:
    """Per-sequence fraction of residues where feature ``feature_id`` fires above ``threshold``."""
    S = int(num_sequences) if num_sequences is not None else len(fd.strings)
    seq_idx_all = np.asarray(fd.seq_idx[:], dtype=np.int64)
    top_idx = np.asarray(fd.top_indices[:])
    top_val = np.asarray(fd.top_values[:])
    mask = top_idx == int(feature_id)
    row_acts = (top_val * mask).sum(axis=1)
    fires = (row_acts > threshold).astype(np.int64)
    total = np.bincount(seq_idx_all, minlength=S)
    fired = np.bincount(seq_idx_all, weights=fires, minlength=S)
    frac = np.zeros(S, dtype=np.float32)
    nz = total > 0
    frac[nz] = (fired[nz] / total[nz]).astype(np.float32)
    return frac


def top_n_sequences(
    fd: FeatureDataset,
    feature_id: int,
    n: int = 20,
    sort_by: str = "peak",
) -> tuple[np.ndarray, np.ndarray]:
    """Top-N sequences for ``feature_id`` ranked by ``sort_by`` ('peak' or 'fraction').

    Returns ``(local_seq_idx[N], score[N])`` in descending order. Sequences
    with zero score are dropped before slicing, so the returned arrays may
    have fewer than ``n`` entries for sparse features.
    """
    if sort_by == "peak":
        scores = per_sequence_peak(fd, feature_id)
    elif sort_by == "fraction":
        scores = per_sequence_fraction_firing(fd, feature_id)
    else:
        raise ValueError(f"sort_by must be 'peak' or 'fraction', got {sort_by!r}")
    nz = np.where(scores > 0)[0]
    if len(nz) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    top = nz[np.argsort(-scores[nz])[:n]]
    return top, scores[top]
