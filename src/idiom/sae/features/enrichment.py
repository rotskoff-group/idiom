"""SAE feature enrichment against a background sequence set.

Uses smoothed log2 odds ratios, a hypergeometric null, and Benjamini-Hochberg FDR.
Signatures exclude boundary-associated features by default.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from idiom.data.records import Record, parse_idr_header, read_fasta
from idiom.sae.features.feature_dataset import FeatureDataset
from idiom.sae.features.signatures import write_signature as write_signature

# Defaults used to build the published signatures
MIN_TOTAL_FIRE = 5
FDR_ALPHA = 1e-3
SMOOTH = 0.5  # Haldane-Anscombe pseudocount added to all four contingency cells
LOG2OR_FLOOR = 1.0
PREV_POS_FLOOR = 0.05

BOUNDARY_EDGE = 2
BOUNDARY_FRAC = 0.5
BOUNDARY_TOP_WINDOWS = 80

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def feature_counts(feature_dir, keep=None) -> tuple[np.ndarray, int]:
    """Count sequences with at least one strictly positive activation per feature.

    Args:
        feature_dir (str | Path): A feature dataset directory.
        keep (Iterable[int] | None): Restrict the count to these sequence indices, or None for all.

    Returns:
        Per-feature sequence counts of length num_latents, and the number of sequences counted.
    """
    fd = feature_dir if isinstance(feature_dir, FeatureDataset) else FeatureDataset(feature_dir)
    ti, tv, si = fd.top_indices, fd.top_values, np.asarray(fd.seq_idx, dtype=np.int64)
    num_latents = fd.num_latents

    if keep is not None:
        keep = np.asarray(sorted(set(fd._sequence_id(k) for k in keep)), dtype=np.int64)
        m = np.isin(si, keep)
        ti, tv, si = ti[m], tv[m], si[m]
        n_seq = len(keep)
    else:
        n_seq = fd.n_seqs

    if not si.size:
        return np.zeros(num_latents), n_seq
    # count DISTINCT (feature, sequence) pairs, i.e. max-pool each feature over each sequence
    big = int(si.max()) + 1
    active = tv.ravel() > 0
    feats = ti.ravel()[active].astype(np.int64)
    seqs = np.repeat(si, ti.shape[1])[active]
    pairs = np.unique(feats * big + seqs)
    return np.bincount(pairs // big, minlength=num_latents).astype(float), n_seq


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in input order, clipped to [0, 1]."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def _two_sided_p(z: np.ndarray) -> np.ndarray:
    """Return the two-sided normal p-value for each z, as erfc(|z| / sqrt(2))."""
    return np.array([math.erfc(abs(float(v)) / math.sqrt(2.0)) for v in z])


def enrich(
    a: np.ndarray,
    n_pos: int,
    b: np.ndarray,
    n_neg: int,
    num_latents: int,
    *,
    smooth: float = SMOOTH,
    min_total_fire: int = MIN_TOTAL_FIRE,
) -> dict:
    """Score every feature for over-representation in the positive set against the background.

    Features whose pooled firing count is below min_total_fire are marked inactive and excluded
    from the FDR correction, leaving their fdr entry NaN.

    Args:
        a: Per-feature count of positive sequences in which the feature fires.
        n_pos: Number of positive sequences.
        b: Per-feature count of background sequences in which the feature fires.
        n_neg: Number of background sequences.
        num_latents: Total number of SAE latents.
        smooth: Pseudocount added to all four contingency cells.
        min_total_fire: Minimum pooled firing count for a feature to be tested.

    Returns:
        A dictionary containing scalar n_pos and n_neg and per-feature arrays a, b,
        log2or, z, p, fdr, active, prev_pos, and prev_neg.
    """
    if (
        n_pos < 1
        or n_neg < 1
        or num_latents < 1
        or min_total_fire < 1
        or not np.isfinite(smooth)
        or smooth <= 0
    ):
        raise ValueError("Positive sample counts, latent count, firing cutoff, and smoothing are required")
    for counts, size in ((a, n_pos), (b, n_neg)):
        values = np.asarray(counts)
        if (
            values.ndim != 1
            or len(values) > num_latents
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > size)
        ):
            raise ValueError("Feature counts must be finite and between zero and the sample count")
    a = np.pad(np.asarray(a, float), (0, num_latents - len(a)))
    b = np.pad(np.asarray(b, float), (0, num_latents - len(b)))
    total = n_pos + n_neg
    k = a + b

    log2or = np.log2(((a + smooth) * (n_neg - b + smooth)) / ((b + smooth) * (n_pos - a + smooth)))
    mu = n_pos * k / total
    var = k * (total - k) * n_pos * (total - n_pos) / (total**2 * (total - 1))
    z = np.where(var > 0, (a - mu) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    p = _two_sided_p(z)

    active = k >= min_total_fire
    fdr = np.full(num_latents, np.nan)
    if active.any():
        fdr[active] = bh_fdr(p[active])
    return dict(
        a=a,
        b=b,
        n_pos=n_pos,
        n_neg=n_neg,
        log2or=log2or,
        z=z,
        p=p,
        fdr=fdr,
        active=active,
        prev_pos=a / max(n_pos, 1),
        prev_neg=b / max(n_neg, 1),
    )


def enriched_mask(
    result: dict,
    *,
    fdr_alpha: float = FDR_ALPHA,
    log2or_floor: float = LOG2OR_FLOOR,
    prev_pos_floor: float = PREV_POS_FLOOR,
) -> np.ndarray:
    """Return a boolean mask of the features passing the FDR, odds-ratio, and prevalence cutoffs.

    Args:
        result: Output of enrich.
        fdr_alpha: FDR ceiling.
        log2or_floor: Minimum log2 odds ratio.
        prev_pos_floor: Minimum prevalence in the positive set.

    Returns:
        A boolean mask over all features.
    """
    if not 0 < fdr_alpha <= 1 or not 0 <= prev_pos_floor <= 1 or not np.isfinite(log2or_floor):
        raise ValueError("Invalid FDR, prevalence, or odds-ratio cutoff")
    return (
        (result["fdr"] < fdr_alpha)
        & (result["log2or"] >= log2or_floor)
        & (result["prev_pos"] >= prev_pos_floor)
    )


def boundary_features(
    feature_dir,
    feature_ids,
    *,
    edge: int = BOUNDARY_EDGE,
    frac_thresh: float = BOUNDARY_FRAC,
    top_windows: int = BOUNDARY_TOP_WINDOWS,
) -> set[int]:
    """Identify features concentrated near the ends of stored FIM sequences.

    Args:
        feature_dir (str | Path): A feature dataset directory.
        feature_ids (Iterable[int]): Candidate features to test.
        edge: Inclusive FIM-index distance from the first or last residue position.
            For example, 2 includes the endpoint and positions up to two indices away.
        frac_thresh: Minimum fraction of top firings at a boundary needed to flag a feature.
        top_windows: Number of top-activating firings per feature to examine.

    Returns:
        The subset of feature_ids that were flagged.
    """
    if edge < 0 or top_windows < 1 or not 0 <= frac_thresh <= 1:
        raise ValueError("Invalid boundary-filter parameters")
    feature_ids = list(feature_ids)
    if not feature_ids:
        return set()
    fd = feature_dir if isinstance(feature_dir, FeatureDataset) else FeatureDataset(feature_dir)
    feature_ids = [fd._feature_id(f) for f in feature_ids]
    ti, tv, si, pi, strings = fd.top_indices, fd.top_values, fd.seq_idx, fd.pos_idx, fd.strings

    cache: dict[int, tuple[int, int]] = {}

    def residue_bounds(seq_index: int) -> tuple[int, int]:
        """Return the first and last residue index within the stored FIM string."""
        if seq_index not in cache:
            s = strings[seq_index]
            aa = [j for j, ch in enumerate(s) if ch in _AA]
            cache[seq_index] = (aa[0], aa[-1]) if aa else (0, -1)
        return cache[seq_index]

    flat = np.asarray(ti).reshape(-1)
    sel = np.isin(flat, feature_ids) & (np.asarray(tv).reshape(-1) > 0)
    rows = np.flatnonzero(sel) // ti.shape[1]
    fids = flat[sel]
    vals = np.asarray(tv).reshape(-1)[sel]

    order = np.argsort(fids, kind="stable")
    fids, vals, rows = fids[order], vals[order], rows[order]
    uniq, starts = np.unique(fids, return_index=True)
    ends = np.append(starts[1:], fids.shape[0])

    flagged: set[int] = set()
    for f, lo_i, hi_i in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        r, v = rows[lo_i:hi_i], vals[lo_i:hi_i]
        top = r[np.argsort(-v, kind="stable")[:top_windows]]
        if not len(top):
            continue
        n_boundary = 0
        for ri in top:
            loc = int(pi[ri])
            first, last = residue_bounds(int(si[ri]))
            if loc <= first + edge or loc >= last - edge:
                n_boundary += 1
        if n_boundary / len(top) >= frac_thresh:
            flagged.add(int(f))
    return flagged


def select_features(
    result: dict,
    *,
    n: int = 30,
    drop_boundary: bool = True,
    feature_dir=None,
    fdr_alpha: float = FDR_ALPHA,
    log2or_floor: float = LOG2OR_FLOOR,
    prev_pos_floor: float = PREV_POS_FLOOR,
) -> dict:
    """Return selected IDs and per-feature enriched, boundary_filtered, selected masks.

    Rank by descending log2 odds ratio, breaking ties by ascending feature ID.
    Boundary exclusions are computed for all passing candidates before taking n.
    """
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    mask = enriched_mask(
        result, fdr_alpha=fdr_alpha, log2or_floor=log2or_floor, prev_pos_floor=prev_pos_floor
    )
    candidates = np.flatnonzero(mask)
    boundary = np.zeros(len(mask), dtype=bool)
    if drop_boundary:
        if feature_dir is None:
            raise ValueError("drop_boundary=True needs feature_dir")
        bad = boundary_features(feature_dir, candidates)
        boundary[list(bad)] = True
    ranked = candidates[np.argsort(-result["log2or"][candidates], kind="stable")]
    ids = [int(f) for f in ranked if not boundary[f]][:n]
    selected = np.zeros(len(mask), dtype=bool)
    selected[ids] = True
    return dict(ids=ids, enriched=mask, boundary_filtered=boundary, selected=selected)


def top_features(result: dict, *, prev_min: float | None = None, **kwargs) -> list[int]:
    """Return selected IDs; prev_min is a compatibility alias for prev_pos_floor."""
    if prev_min is not None:
        if "prev_pos_floor" in kwargs and kwargs["prev_pos_floor"] != prev_min:
            raise ValueError("Specify only one prevalence cutoff")
        kwargs["prev_pos_floor"] = prev_min
    return select_features(result, **kwargs)["ids"]


def save_enrichment(path, result: dict, selection: dict) -> Path:
    """Save a complete numerical result and selection masks for reuse without inference."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        np.savez_compressed(handle, **result, **selection)
    return out


def load_enrichment(path) -> tuple[dict, dict]:
    """Read the result and selection written by save_enrichment (no pickle)."""
    with np.load(path, allow_pickle=False) as data:
        result = {
            key: data[key]
            for key in data.files
            if key not in {"ids", "enriched", "boundary_filtered", "selected"}
        }
        selection = {key: data[key] for key in ("enriched", "boundary_filtered", "selected")}
        selection["ids"] = data["ids"].astype(int).tolist()
    for key in ("n_pos", "n_neg"):
        result[key] = int(result[key])
    return result, selection


def load_sequences(path) -> list[Record]:
    """Read canonical FASTA records, treating unusable IDR spans as the whole sequence.

    Records retain file order.
    """
    out = []
    for header, seq in read_fasta(path):
        try:
            acc, start, end = parse_idr_header(header)
            if not 0 <= start < end <= len(seq):
                raise ValueError
        except ValueError:
            acc, start, end = header.split()[0], 0, len(seq)
        out.append(Record(acc, seq, start, end))
    return out


def length_match(positives, background, *, n, rng, bin_width=20) -> list[Record]:
    """Sample a background whose IDR-length distribution follows the positive set's.

    Allocate bins by largest remainder, then fill undersupplied bins from the remaining
    pool. Return exactly min(n, len(background)) records, sampled without replacement.

    Args:
        positives (list[Record]): Positive records.
        background (list[Record]): Candidate background records.
        n (int): Target background size.
        rng (np.random.Generator): Random source.
        bin_width (int): Length-bin width in residues.

    Returns:
        The sampled background records, without replacement within each bin.
    """

    if not isinstance(n, (int, np.integer)) or n < 0 or not isinstance(bin_width, int) or bin_width < 1:
        raise ValueError("n must be nonnegative and bin_width positive integers")
    positives, background = list(positives), list(background)
    if not positives:
        raise ValueError("At least one positive record is required")
    target = min(n, len(background))
    if target == 0:
        return []

    def length_bin(r):
        return (r.idr_end - r.idr_start) // bin_width

    pools = {}
    for i, record in enumerate(background):
        pools.setdefault(length_bin(record), []).append(i)
    bins, counts = np.unique([length_bin(r) for r in positives], return_counts=True)
    exact = counts / counts.sum() * target
    allocation = np.floor(exact).astype(int)
    remainder = target - allocation.sum()
    allocation[np.argsort(-(exact - allocation), kind="stable")[:remainder]] += 1
    picked = []
    for b, want in zip(bins, allocation):
        pool = pools.get(b, [])
        if pool:
            picked.extend(rng.choice(pool, size=min(want, len(pool)), replace=False).tolist())
    if len(picked) < target:
        rest = np.setdiff1d(np.arange(len(background)), picked)
        picked.extend(rng.choice(rest, size=target - len(picked), replace=False).tolist())
    return [background[i] for i in picked]
