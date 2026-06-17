"""Streamlit viewer for SAE features.

Reads a feature-activation dataset produced by ``build_feature_dataset`` and shows, for a chosen
feature, its top-N activating sequences with per-residue background shading proportional to
activation strength.

The dataset is loaded into RAM once (``FeatureDataset(in_memory=True)``); the per-feature
reductions live on :class:`~idiom.sae.features.feature_activations.FeatureDataset` (a single cached scan for
the ranking, an O(rows-in-sequence) gather per trace via its CSR index), so nothing touches disk on
a rerun. This keeps latency flat as the dataset grows — the cost moves to the one-time load and RAM
footprint (~``N_res * (8*k + 16)`` bytes; e.g. ~3 GB at 11M residues, k=32).

Run:
    streamlit run src/idiom/sae/features/feature_viewer.py -- \\
        --features ./data/feature_dataset_L6
"""

from __future__ import annotations

import argparse
import html

import numpy as np
import streamlit as st

from idiom.sae.features.feature_activations import FeatureDataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="Path to a feature-activation dataset dir")
    args, _ = p.parse_known_args()
    return args


@st.cache_resource
def _load(path: str) -> FeatureDataset:
    """Open the dataset once and pull everything into RAM (CSR index built in the ctor)."""
    return FeatureDataset(path, in_memory=True)


@st.cache_data
def _ranking(_fd: FeatureDataset, num_latents: int):
    """Per-feature global scores (max, total, firing count). Cached for the session."""
    return _fd.feature_ranking()


@st.cache_data
def _stats(_fd: FeatureDataset, feature_id: int):
    """Global max + per-sequence peak and fraction-firing for one feature (cached per feature)."""
    return _fd.feature_stats(feature_id)


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
    fd = _load(args.features)
    n_seqs = fd.n_seqs

    st.set_page_config(layout="wide")
    st.title(f"SAE feature viewer — layer {fd.layer} · region {fd.region}")

    # Order features strongest-first so the viewer opens on the most active feature rather than
    # feature 0; never-firing features are dropped from the navigation.
    fmax, fsum, fcount = _ranking(fd, fd.num_latents)
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

    gmax, peak, frac = _stats(fd, feature_id)
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
        seq = fd.sequence(s)
        pos, acts = fd.trace(s, feature_id)
        full = np.zeros(len(seq), dtype=np.float32)
        full[pos] = acts
        st.markdown(
            f"**#{rank}** · seq_idx={s} · peak={float(full.max()):.3f} · "
            f"frac_firing={(full > 0).mean():.3f} · len={len(seq)}"
        )
        st.markdown(_shade(seq, full, gmax), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
