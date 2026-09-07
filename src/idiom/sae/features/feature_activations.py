"""Read and reduce per-residue SAE feature datasets.

Files:
    top_indices.npy: int32 [N_res, k] selected latent ids.
    top_values.npy: float32 [N_res, k] activations.
    seq_idx.npy: int32 [N_res] sequence indices into strings.json.
    pos_idx.npy: int32 [N_res] positions in FIM strings, excluding START.
    strings.json: FIM strings in sequence order.
    meta.json: k, num_latents, layer, region, and fim_mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatureDataset:
    """Reader over a feature-activation dataset directory.

    Attributes:
        path (Path): The dataset directory.
        in_memory (bool): Whether the arrays were loaded into RAM rather than memory-mapped.
        top_indices (np.ndarray): Selected latent ids, shape [N_res, k].
        top_values (np.ndarray): Their activations, shape [N_res, k].
        seq_idx (np.ndarray): Sequence index per row, shape [N_res].
        pos_idx (np.ndarray): Position within that sequence's string, shape [N_res].
        strings (list[str]): The FIM string per sequence.
        k (int): Latents kept per residue.
        num_latents (int): Total number of SAE latents.
        layer (int): The layer the activations came from.
        region (str): The residue region the dataset covers.
        n_seqs (int): Number of sequences.
    """

    def __init__(self, path: str | Path, in_memory: bool = False):
        """Open a dataset directory and build the per-sequence row index.

        Args:
            path: The dataset directory.
            in_memory: If True, load the arrays into RAM; otherwise memory-map them.
        """
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

        # CSR-style grouping of residue rows by their sequence: the rows for local sequence s
        # are _order[_offsets[s]:_offsets[s + 1]]. Stable sort keeps row order deterministic.
        seq = np.asarray(self.seq_idx[:], dtype=np.int64)
        self.n_seqs = len(self.strings)
        self._order = np.argsort(seq, kind="stable").astype(np.int64)
        self._offsets = np.zeros(self.n_seqs + 1, dtype=np.int64)
        np.cumsum(np.bincount(seq, minlength=self.n_seqs), out=self._offsets[1:])
        self._ranking: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def sequence(self, local_seq_idx: int) -> str:
        """Return the stored FIM string for local_seq_idx."""
        s = self.strings[int(local_seq_idx)]
        return s.decode("utf-8") if isinstance(s, bytes) else str(s)

    # Rows processed per pass when streaming a memory-mapped dataset. Bounds peak RAM to
    # ~CHUNK_ROWS * k * 8 bytes (e.g. ~256 MB at 1M rows, k=32) instead of copying the whole
    # [N_res, k] arrays into RAM (which arr[:] does even on a memmap).
    CHUNK_ROWS = 1_000_000

    def _row_chunks(self):
        """Yield (start_row, top_indices, top_values) chunks, in one piece when held in memory."""
        n = self.top_indices.shape[0]
        if self.in_memory:
            yield 0, np.asarray(self.top_indices), np.asarray(self.top_values)
            return
        for s in range(0, n, self.CHUNK_ROWS):
            e = min(s + self.CHUNK_ROWS, n)
            yield s, np.asarray(self.top_indices[s:e]), np.asarray(self.top_values[s:e])

    # --- reductions ---
    def row_activations(self, feature_id: int) -> np.ndarray:
        """Return float32 [N_res] activations, zero where feature_id was not selected."""
        f = int(feature_id)
        out = np.zeros(self.top_indices.shape[0], dtype=np.float32)
        for s, ti, tv in self._row_chunks():
            out[s:s + ti.shape[0]] = (tv * (ti == f)).sum(axis=1)
        return out

    def feature_ranking(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-feature maximum, total, and firing count over the whole dataset.

        Zero-valued selections are ignored, and the result is cached.

        Returns:
            The per-latent maximum activation, total activation, and firing count, each of shape
            [num_latents].
        """
        if self._ranking is not None:
            return self._ranking
        fsum = np.zeros(self.num_latents, dtype=np.float64)
        fcount = np.zeros(self.num_latents, dtype=np.int64)
        fmax = np.zeros(self.num_latents, dtype=np.float32)
        for _s, ti, tv in self._row_chunks():
            fi = np.asarray(ti).reshape(-1)
            fv = np.asarray(tv).reshape(-1)
            keep = fv > 0
            fi = fi[keep].astype(np.int64)
            fv = fv[keep]
            if not fi.size:
                continue
            fsum += np.bincount(fi, weights=fv, minlength=self.num_latents)
            fcount += np.bincount(fi, minlength=self.num_latents)
            srt = np.argsort(fi, kind="stable")
            uniq, start = np.unique(fi[srt], return_index=True)
            np.maximum.at(fmax, uniq, np.maximum.reduceat(fv[srt], start).astype(np.float32))
        self._ranking = (fmax, fsum.astype(np.float32), fcount)
        return self._ranking

    def feature_stats(self, feature_id: int) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (global_max, sequence_peaks, sequence_firing_fractions) for feature_id.

        Both per-sequence arrays have shape [n_seqs]; firing means activation > 0.
        """
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
        """Return sorted (FIM_positions, activations) for one sequence and feature.

        Unselected features have zero activation; sequences without rows return empty arrays.
        """
        s = int(local_seq_idx)
        rows = self._order[self._offsets[s] : self._offsets[s + 1]]
        if rows.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        pos = np.asarray(self.pos_idx[rows], dtype=np.int64)
        hit = np.asarray(self.top_indices[rows]) == int(feature_id)
        acts = (np.asarray(self.top_values[rows]) * hit).sum(axis=1)
        order_p = np.argsort(pos)
        return pos[order_p], acts[order_p].astype(np.float32)

    def top_sequences(
        self, feature_id: int, n: int = 20, sort_by: str = "peak"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the highest-scoring sequences for one feature.

        Sequences scoring zero are dropped, so fewer than n may be returned.

        Args:
            feature_id: The latent to rank sequences for.
            n: Maximum sequences to return.
            sort_by: Ranking key, "peak" or "fraction".

        Returns:
            The sequence indices and their scores, in descending order of score.

        Raises:
            ValueError: If sort_by is neither "peak" nor "fraction".
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
