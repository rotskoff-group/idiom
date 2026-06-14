"""Streamlit viewer for SAE features.

Reads a feature-activation dataset produced by the feature-dataset builder and shows, for a
chosen feature, its top-N activating sequences with per-residue background shading
proportional to activation strength.

The whole HDF5 is loaded into RAM once (``FeatureDataset(in_memory=True)``) and a
CSR-style per-sequence row index is built so that, on each Streamlit rerun,
nothing touches disk: per-feature reductions are a single cached in-RAM scan and
each rendered sequence's trace is an O(residues-in-sequence) gather. This keeps
latency flat as the HDF5 grows — the cost moves to the one-time load and RAM
footprint (~``N_res * (8*k + 16)`` bytes; e.g. ~3 GB at 11M residues, k=32).

Run:
    streamlit run src/idiom/sae/feature_viewer.py -- \\
        --features ./data/feature_dataset_L6
"""

from __future__ import annotations

import argparse
import html

import numpy as np
import streamlit as st

from idiom.sae.feature_activations import FeatureDataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="Path to feature_dataset_L*.h5")
    args, _ = p.parse_known_args()
    return args


@st.cache_resource
def _load(path: str):
    """Open the HDF5 once, pull everything into RAM, and index rows by sequence.

    Streamlit reruns the whole script on every widget change, so per-rerun h5
    reads would dominate latency. We materialize the per-residue arrays in RAM
    and precompute a CSR-style index (``order``/``offsets``) grouping rows by
    sequence: the rows for local sequence ``s`` are ``order[offsets[s]:offsets[s+1]]``.
    That turns the per-sequence trace from an O(N_res) scan into an O(rows-in-seq)
    slice, independent of HDF5 size.
    """
    fd = FeatureDataset(path, in_memory=True)
    top_indices = fd.top_indices
    top_values = fd.top_values
    seq_idx_all = fd.seq_idx.astype(np.int64)
    pos_idx_all = fd.pos_idx.astype(np.int64)
    strings = [
        s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in fd.strings
    ]
    n_seqs = len(strings)

    # CSR-style grouping of residue rows by their sequence. Stable sort keeps
    # row order within a sequence deterministic; offsets are cumulative counts.
    order = np.argsort(seq_idx_all, kind="stable").astype(np.int64)
    counts = np.bincount(seq_idx_all, minlength=n_seqs)
    offsets = np.zeros(n_seqs + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    return fd, top_indices, top_values, seq_idx_all, pos_idx_all, strings, order, offsets


@st.cache_data
def _feature_stats(feature_id: int, _top_idx, _top_val, _seq_idx_all, n_seqs: int):
    """Global max + per-sequence peak and fraction-firing, in one cached scan.

    Cached per ``feature_id`` so the full O(N_res*k) pass runs once per feature
    rather than on every rerun. Underscored array args are excluded from the
    cache key (they are stable for the session).
    """
    mask = _top_idx == feature_id
    row_acts = (_top_val * mask).sum(axis=1)
    gmax = float(row_acts.max()) if row_acts.size else 0.0

    peak = np.zeros(n_seqs, dtype=np.float32)
    np.maximum.at(peak, _seq_idx_all, row_acts)

    total = np.bincount(_seq_idx_all, minlength=n_seqs)
    fired = np.bincount(
        _seq_idx_all, weights=(row_acts > 0).astype(np.float64), minlength=n_seqs
    )
    frac = np.zeros(n_seqs, dtype=np.float32)
    nz = total > 0
    frac[nz] = (fired[nz] / total[nz]).astype(np.float32)

    return gmax, peak, frac


@st.cache_data
def _feature_ranking(_top_idx, _top_val, num_latents: int):
    """Per-feature global scores (max, total, firing count) over the whole dataset.

    One cached O(N_res*k) pass used to order features strongest-first. Padded
    (zero-value) top-k slots are ignored. Underscored array args are excluded
    from the cache key; ``num_latents`` keys the (single-dataset) session.
    """
    flat_idx = _top_idx.reshape(-1)
    flat_val = _top_val.reshape(-1)
    keep = flat_val > 0
    flat_idx = flat_idx[keep].astype(np.int64)
    flat_val = flat_val[keep]

    fsum = np.bincount(flat_idx, weights=flat_val, minlength=num_latents).astype(np.float32)
    fcount = np.bincount(flat_idx, minlength=num_latents).astype(np.int64)

    # Per-feature max via sort + segmented reduce (avoids the slow unbuffered
    # np.maximum.at on the full flattened top-k array).
    fmax = np.zeros(num_latents, dtype=np.float32)
    if flat_idx.size:
        srt = np.argsort(flat_idx, kind="stable")
        sidx = flat_idx[srt]
        uniq, start = np.unique(sidx, return_index=True)
        fmax[uniq] = np.maximum.reduceat(flat_val[srt], start)
    return fmax, fsum, fcount


def _trace(s, feature_id, top_idx, top_val, pos_idx_all, order, offsets):
    """Per-residue (positions, activations) for one sequence, gathered in RAM.

    Uses the CSR index to grab just this sequence's rows; no full-array scan.
    """
    rows = order[offsets[s] : offsets[s + 1]]
    if rows.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    pos = pos_idx_all[rows]
    sub_idx = top_idx[rows]
    sub_val = top_val[rows]
    acts = (sub_val * (sub_idx == feature_id)).sum(axis=1)
    order_p = np.argsort(pos)
    return pos[order_p], acts[order_p]


def _shade(seq: str, acts: np.ndarray, gmax: float) -> str:
    """Render a sequence as one <span> per residue with alpha ∝ activation."""
    parts = []
    for ch, a in zip(seq, acts):
        alpha = float(a) / gmax if gmax > 0 else 0.0
        parts.append(
            f"<span style='background:rgba(255,140,0,{alpha:.3f})'>"
            f"{html.escape(ch)}</span>"
        )
    return (
        "<div style='font-family:monospace;font-size:13px;"
        "word-break:break-all;line-height:1.4'>" + "".join(parts) + "</div>"
    )


def main() -> None:
    args = _parse_args()
    fd, top_idx, top_val, seq_idx_all, pos_idx_all, strings, order, offsets = _load(
        args.features
    )
    n_seqs = len(strings)

    st.set_page_config(layout="wide")
    st.title(f"SAE feature viewer — layer {fd.layer}")

    # Order features strongest-first so the viewer opens on the most active
    # feature rather than feature 0; never-firing features are dropped from the
    # navigation (a tiny, narrow dataset leaves most of the dictionary dark).
    fmax, fsum, fcount = _feature_ranking(top_idx, top_val, fd.num_latents)
    metric_arrays = {
        "max activation": fmax,
        "total activation": fsum,
        "# firings": fcount.astype(np.float32),
    }

    col1, col2, col3, col4 = st.columns([1.3, 1, 1, 1])
    rank_metric = col1.selectbox("Order features by", list(metric_arrays), index=0)
    fscore = metric_arrays[rank_metric]
    n_alive = int((fscore > 0).sum())
    feature_order = np.argsort(-fscore, kind="stable")[:n_alive]

    rank = int(col2.number_input(
        "Rank (0 = strongest)", min_value=0, max_value=max(n_alive - 1, 0), value=0, step=1
    ))
    override = int(col3.number_input(
        "Jump to feature (-1 = use rank)", min_value=-1, max_value=fd.num_latents - 1, value=-1, step=1
    ))
    n_top = int(col4.slider("Top sequences", 5, 200, 100, step=5))
    sort_by = st.radio("Sort sequences by", ["peak", "fraction"], horizontal=True)

    if n_alive == 0:
        st.warning("No features fire in this dataset.")
        return
    feature_id = override if override >= 0 else int(feature_order[rank])

    gmax, peak, frac = _feature_stats(feature_id, top_idx, top_val, seq_idx_all, n_seqs)
    n_firing = int((peak > 0).sum())
    where = np.where(feature_order == feature_id)[0]
    rank_str = f"{int(where[0]):,}" if where.size else "dead"
    st.caption(
        f"feature {feature_id}: rank {rank_str} / {n_alive:,} by {rank_metric} | "
        f"max={fmax[feature_id]:.3f} · total={fsum[feature_id]:.1f} · "
        f"firings={int(fcount[feature_id]):,} | "
        f"global max = {gmax:.3f} | {n_firing:,} / {n_seqs:,} sequences fire"
    )

    scores = peak if sort_by == "peak" else frac
    nz = np.where(scores > 0)[0]
    if len(nz) == 0:
        st.warning("Feature is dead in this dataset.")
        return
    top_seqs = nz[np.argsort(-scores[nz])[:n_top]]

    for rank, s in enumerate(top_seqs.tolist(), 1):
        seq = strings[s]
        pos, acts = _trace(s, feature_id, top_idx, top_val, pos_idx_all, order, offsets)
        full = np.zeros(len(seq), dtype=np.float32)
        full[pos] = acts
        st.markdown(
            f"**#{rank}** · seq_idx={s} · peak={float(full.max()):.3f} · "
            f"frac_firing={(full > 0).mean():.3f} · len={len(seq)}"
        )
        st.markdown(_shade(seq, full, gmax), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
