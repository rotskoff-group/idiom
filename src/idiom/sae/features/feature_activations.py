"""Reader + reductions over the feature-activation dataset produced by ``build_feature_dataset``.

A directory (no h5) of top-k sparse SAE activations per residue, with ``(seq_idx, pos_idx)``
joins back to the raw FIM sequence strings::

    top_indices.npy  int32[N_res, k]   which latents fired at each residue
    top_values.npy   float32[N_res, k] their activations
    seq_idx.npy      int32[N_res]      index into strings
    pos_idx.npy      int32[N_res]      position within strings[s] (the FIM string)
    strings.json     list[str]         raw FIM `1{prefix}3{suffix}2{IDR}` per sequence
    meta.json        {k, num_latents, layer, region}

:class:`FeatureDataset` is the single home for the per-feature reductions the viewer / annotator /
concept correlator use to slice that artifact — no streaming, no GPU. It builds a CSR-style index
(``_order`` / ``_offsets``) grouping residue rows by sequence once, so a per-sequence trace is an
O(rows-in-sequence) gather instead of a full scan; the global per-feature ranking is one cached
pass. All reductions live here so the viewer and the tests exercise the same code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatureDataset:
    """Reader over a feature-activation dataset directory (``.npy`` + ``.json``).

    By default the arrays stay memory-mapped (slicing hits disk); pass ``in_memory=True`` to pull
    them fully into RAM, which is what the Streamlit viewer wants since it reslices on every widget
    change. Either way the reductions below are the single implementation.
    """

    def __init__(self, path: str | Path, in_memory: bool = False):
        self.path = Path(path)
        self.in_memory = in_memory
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
        self.region = str(meta.get("region", "all"))

        # CSR-style grouping of residue rows by their sequence: the rows for local sequence ``s``
        # are ``_order[_offsets[s]:_offsets[s + 1]]``. Stable sort keeps row order deterministic.
        seq = np.asarray(self.seq_idx[:], dtype=np.int64)
        self.n_seqs = len(self.strings)
        self._order = np.argsort(seq, kind="stable").astype(np.int64)
        self._offsets = np.zeros(self.n_seqs + 1, dtype=np.int64)
        np.cumsum(np.bincount(seq, minlength=self.n_seqs), out=self._offsets[1:])
        self._ranking: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def sequence(self, local_seq_idx: int) -> str:
        s = self.strings[int(local_seq_idx)]
        return s.decode("utf-8") if isinstance(s, bytes) else str(s)

    # --- reductions ---
    def row_activations(self, feature_id: int) -> np.ndarray:
        """Per-residue activation of ``feature_id`` over all rows (0 where not in top-k)."""
        top_idx = np.asarray(self.top_indices[:])
        top_val = np.asarray(self.top_values[:])
        return (top_val * (top_idx == int(feature_id))).sum(axis=1)

    def feature_ranking(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-feature ``(max, total, count)`` over the whole dataset (one cached pass).

        Padded (zero-value) top-k slots are ignored. Used to order features strongest-first.
        """
        if self._ranking is not None:
            return self._ranking
        flat_idx = np.asarray(self.top_indices[:]).reshape(-1)
        flat_val = np.asarray(self.top_values[:]).reshape(-1)
        keep = flat_val > 0
        flat_idx = flat_idx[keep].astype(np.int64)
        flat_val = flat_val[keep]

        fsum = np.bincount(flat_idx, weights=flat_val, minlength=self.num_latents).astype(np.float32)
        fcount = np.bincount(flat_idx, minlength=self.num_latents).astype(np.int64)
        # Per-feature max via sort + segmented reduce (faster than np.maximum.at on the flat array).
        fmax = np.zeros(self.num_latents, dtype=np.float32)
        if flat_idx.size:
            srt = np.argsort(flat_idx, kind="stable")
            uniq, start = np.unique(flat_idx[srt], return_index=True)
            fmax[uniq] = np.maximum.reduceat(flat_val[srt], start)
        self._ranking = (fmax, fsum, fcount)
        return self._ranking

    def feature_stats(self, feature_id: int) -> tuple[float, np.ndarray, np.ndarray]:
        """``(global_max, peak[n_seqs], fraction_firing[n_seqs])`` for ``feature_id``."""
        seq = np.asarray(self.seq_idx[:], dtype=np.int64)
        row_acts = self.row_activations(feature_id)
        gmax = float(row_acts.max()) if row_acts.size else 0.0

        peak = np.zeros(self.n_seqs, dtype=np.float32)
        np.maximum.at(peak, seq, row_acts)

        total = np.bincount(seq, minlength=self.n_seqs)
        fired = np.bincount(seq, weights=(row_acts > 0).astype(np.float64), minlength=self.n_seqs)
        frac = np.zeros(self.n_seqs, dtype=np.float32)
        nz = total > 0
        frac[nz] = (fired[nz] / total[nz]).astype(np.float32)
        return gmax, peak, frac

    def trace(self, local_seq_idx: int, feature_id: int) -> tuple[np.ndarray, np.ndarray]:
        """``(positions, activations)`` of ``feature_id`` across one sequence, sorted by position.

        Uses the CSR index to gather just this sequence's rows. Residues where the feature was not
        in the top-k contribute zero.
        """
        s = int(local_seq_idx)
        rows = self._order[self._offsets[s] : self._offsets[s + 1]]
        if rows.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        pos = np.asarray(self.pos_idx[rows], dtype=np.int64)
        acts = (np.asarray(self.top_values[rows]) * (np.asarray(self.top_indices[rows]) == int(feature_id))).sum(axis=1)
        order_p = np.argsort(pos)
        return pos[order_p], acts[order_p].astype(np.float32)

    def top_sequences(
        self, feature_id: int, n: int = 20, sort_by: str = "peak"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-N sequences for ``feature_id`` ranked by ``sort_by`` ('peak' or 'fraction').

        Returns ``(local_seq_idx[N], score[N])`` descending. Zero-score sequences are dropped, so
        the arrays may be shorter than ``n`` for sparse features.
        """
        _, peak, frac = self.feature_stats(feature_id)
        if sort_by == "peak":
            scores = peak
        elif sort_by == "fraction":
            scores = frac
        else:
            raise ValueError(f"sort_by must be 'peak' or 'fraction', got {sort_by!r}")
        nz = np.where(scores > 0)[0]
        if len(nz) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        top = nz[np.argsort(-scores[nz])[:n]]
        return top, scores[top]
