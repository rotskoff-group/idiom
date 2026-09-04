"""Testing which SAE features are over-represented in a set of sequences against a background.

A feature fires in a sequence if the SAE selects it at any of that sequence's residues, counted
once per sequence. From the positive and background firing counts, enrich computes a
Haldane-Anscombe log2 odds ratio, standardizes it against a hypergeometric null, converts that to a
two-sided p-value, and controls the false discovery rate with Benjamini-Hochberg. A feature is
enriched when its FDR, log2 odds ratio, and prevalence all pass the module's thresholds, and
top_features keeps the strongest as a signature.

boundary_features identifies features whose strongest firings sit at an IDR's first or last
residues, detecting the excision boundary rather than a motif; top_features drops them by default.

load_sequences reads a FASTA whether or not its headers carry an IDR span, and length_match samples
a background whose length distribution follows the positive set's.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from idiom.data.io import Record, parse_idr_header, read_fasta

# Enrichment thresholds (the published signatures were built with exactly these).
MIN_TOTAL_FIRE = 5      # a feature must fire in >= this many sequences overall to be tested
FDR_ALPHA = 1e-3        # Benjamini-Hochberg false discovery rate ceiling
SMOOTH = 0.5            # Haldane-Anscombe pseudocount added to all four contingency cells
LOG2OR_FLOOR = 1.0      # minimum log2 odds ratio
PREV_POS_FLOOR = 0.05   # minimum fraction of the positive set in which the feature fires

# Boundary-artifact detection (features that fire at an IDR's excision points, not on a motif).
BOUNDARY_EDGE = 2       # residues from either end that count as "at the boundary"
BOUNDARY_FRAC = 0.5     # flag if >= this fraction of a feature's top windows are at a boundary
BOUNDARY_TOP_WINDOWS = 80   # top-activating firings per feature examined

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def feature_counts(feature_dir, keep=None) -> tuple[np.ndarray, int]:
    """Count, per feature, the number of sequences in which it fires at least once.

    Args:
        feature_dir (str | Path): A feature dataset directory.
        keep (Iterable[int] | None): Restrict the count to these sequence indices, or None for all.

    Returns:
        tuple[np.ndarray, int]: Per-feature sequence counts of length num_latents, and the number
            of sequences counted.
    """
    d = Path(feature_dir)
    ti = np.load(d / "top_indices.npy")
    si = np.load(d / "seq_idx.npy").astype(np.int64)
    num_latents = int(json.loads((d / "meta.json").read_text())["num_latents"])

    if keep is not None:
        keep = np.asarray(sorted(set(int(k) for k in keep)))
        m = np.isin(si, keep)
        ti, si = ti[m], si[m]
        n_seq = len(keep)
    else:
        n_seq = int(si.max()) + 1 if si.size else 0

    if not si.size:
        return np.zeros(num_latents), n_seq
    # count DISTINCT (feature, sequence) pairs, i.e. max-pool each feature over each sequence
    big = int(si.max()) + 1
    feats = ti.ravel().astype(np.int64)
    seqs = np.repeat(si, ti.shape[1])
    pairs = np.unique(feats * big + seqs)
    return np.bincount(pairs // big, minlength=num_latents).astype(float), n_seq


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg adjusted p-values.

    Args:
        p (np.ndarray): Raw p-values.

    Returns:
        np.ndarray: Adjusted p-values in the input order, each clipped to [0, 1] and
            non-decreasing in the raw p-value.
    """
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


def enrich(a: np.ndarray, n_pos: int, b: np.ndarray, n_neg: int, num_latents: int, *,
           smooth: float = SMOOTH, min_total_fire: int = MIN_TOTAL_FIRE) -> dict:
    """Score every feature for over-representation in the positive set against the background.

    Features whose pooled firing count is below min_total_fire are marked inactive and excluded
    from the FDR correction, leaving their fdr entry NaN.

    Args:
        a (np.ndarray): Per-feature count of positive sequences in which the feature fires.
        n_pos (int): Number of positive sequences.
        b (np.ndarray): Per-feature count of background sequences in which the feature fires.
        n_neg (int): Number of background sequences.
        num_latents (int): Total number of SAE latents.
        smooth (float): Pseudocount added to all four contingency cells.
        min_total_fire (int): Minimum pooled firing count for a feature to be tested.

    Returns:
        dict: The counts a and b, the sizes n_pos and n_neg, and the per-feature arrays log2or, z,
            p, fdr, active, prev_pos, and prev_neg.
    """
    a = np.pad(np.asarray(a, float), (0, num_latents - len(a)))
    b = np.pad(np.asarray(b, float), (0, num_latents - len(b)))
    total = n_pos + n_neg
    k = a + b

    log2or = np.log2(((a + smooth) * (n_neg - b + smooth)) / ((b + smooth) * (n_pos - a + smooth)))
    mu = n_pos * k / total
    var = k * (total - k) * n_pos * (total - n_pos) / (total ** 2 * (total - 1))
    z = np.where(var > 0, (a - mu) / np.sqrt(np.maximum(var, 1e-12)), 0.0)
    p = _two_sided_p(z)

    active = k >= min_total_fire
    fdr = np.full(num_latents, np.nan)
    if active.any():
        fdr[active] = bh_fdr(p[active])
    return dict(a=a, b=b, n_pos=n_pos, n_neg=n_neg, log2or=log2or, z=z, p=p, fdr=fdr,
                active=active, prev_pos=a / max(n_pos, 1), prev_neg=b / max(n_neg, 1))


def enriched_mask(result: dict, *, fdr_alpha: float = FDR_ALPHA,
                  log2or_floor: float = LOG2OR_FLOOR,
                  prev_pos_floor: float = PREV_POS_FLOOR) -> np.ndarray:
    """Return a boolean mask of the features passing the FDR, odds-ratio, and prevalence cutoffs.

    Args:
        result (dict): Output of enrich.
        fdr_alpha (float): FDR ceiling.
        log2or_floor (float): Minimum log2 odds ratio.
        prev_pos_floor (float): Minimum prevalence in the positive set.

    Returns:
        np.ndarray: A boolean mask over all features.
    """
    return ((result["fdr"] < fdr_alpha)
            & (result["log2or"] >= log2or_floor)
            & (result["prev_pos"] >= prev_pos_floor))


def boundary_features(feature_dir, feature_ids, *, edge: int = BOUNDARY_EDGE,
                      frac_thresh: float = BOUNDARY_FRAC,
                      top_windows: int = BOUNDARY_TOP_WINDOWS) -> set[int]:
    """Identify features whose strongest firings sit at the first or last residues of a sequence.

    For each candidate feature, the top_windows highest-activating firings are examined and the
    feature is flagged if at least frac_thresh of them fall within edge residues of either end.

    Args:
        feature_dir (str | Path): A feature dataset directory.
        feature_ids (Iterable[int]): Candidate features to test.
        edge (int): Number of residues from either end that count as a boundary.
        frac_thresh (float): Fraction of top firings at a boundary above which a feature is
            flagged.
        top_windows (int): Number of top-activating firings per feature to examine.

    Returns:
        set[int]: The subset of feature_ids that were flagged.
    """
    feature_ids = [int(f) for f in feature_ids]
    if not feature_ids:
        return set()
    d = Path(feature_dir)
    ti = np.load(d / "top_indices.npy", mmap_mode="r")
    tv = np.load(d / "top_values.npy", mmap_mode="r")
    si = np.load(d / "seq_idx.npy")
    pi = np.load(d / "pos_idx.npy")
    strings = json.loads((d / "strings.json").read_text())

    cache: dict[int, tuple[int, int]] = {}

    def residue_bounds(seq_index: int) -> tuple[int, int]:
        """Return the first and last residue index within the stored FIM string."""
        if seq_index not in cache:
            s = strings[seq_index]
            aa = [j for j, ch in enumerate(s) if ch in _AA]
            cache[seq_index] = (aa[0], aa[-1]) if aa else (0, -1)
        return cache[seq_index]

    want = np.zeros(max(int(np.asarray(ti).max()), max(feature_ids)) + 1, dtype=bool)
    want[feature_ids] = True
    flat = np.asarray(ti).reshape(-1)
    sel = want[flat]
    rows = np.repeat(np.arange(ti.shape[0]), ti.shape[1])[sel]
    fids = flat[sel]
    vals = np.asarray(tv).reshape(-1)[sel]

    order = np.argsort(fids, kind="stable")
    fids, vals, rows = fids[order], vals[order], rows[order]
    uniq, starts = np.unique(fids, return_index=True)
    ends = np.append(starts[1:], fids.shape[0])

    flagged: set[int] = set()
    for f, lo_i, hi_i in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        r, v = rows[lo_i:hi_i], vals[lo_i:hi_i]
        top = r[np.argsort(-v)[:top_windows]]
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


def top_features(result: dict, *, n: int = 30, prev_min: float = PREV_POS_FLOOR,
                 drop_boundary: bool = True, feature_dir=None, **mask_kwargs) -> list[int]:
    """Select a signature as the top n enriched features, ranked by log2 odds ratio.

    Args:
        result (dict): Output of enrich.
        n (int): Maximum number of features to keep.
        prev_min (float): Minimum prevalence in the positive set.
        drop_boundary (bool): If True, remove boundary features before truncating to n.
        feature_dir (str | Path | None): Feature dataset used to detect boundary features;
            required when drop_boundary is True.
        **mask_kwargs: Forwarded to enriched_mask as fdr_alpha, log2or_floor, and prev_pos_floor.

    Returns:
        list[int]: Feature ids in descending order of log2 odds ratio, fewer than n if the
            enriched pool is smaller.

    Raises:
        ValueError: If drop_boundary is True and feature_dir is None.
    """
    mask = enriched_mask(result, **mask_kwargs) & (result["prev_pos"] >= prev_min)
    ids = np.where(mask)[0]
    ids = ids[np.argsort(-result["log2or"][ids])]          # rank by log2 odds ratio
    ranked = [int(f) for f in ids]

    if drop_boundary:
        if feature_dir is None:
            raise ValueError("drop_boundary=True needs feature_dir (the dataset to judge against)")
        bad = boundary_features(feature_dir, ranked)
        ranked = [f for f in ranked if f not in bad]
    return ranked[:n]


def write_signature(path, signatures: dict[str, list[int]], *, case: str = "top30",
                    provenance: dict | None = None) -> Path:
    """Write signatures to a JSON file in the format the SAE feature reward reads.

    An existing file is read and updated, and the named case is replaced.

    Args:
        path (str | Path): Output JSON path.
        signatures (dict[str, list[int]]): Signature name to feature ids.
        case (str): Case name to store the signatures under.
        provenance (dict | None): Notes merged into the file's "_provenance" entry.

    Returns:
        Path: The written path.
    """
    out = Path(path)
    blob = json.loads(out.read_text()) if out.exists() else {}
    blob[case] = {k: [int(i) for i in v] for k, v in signatures.items()}
    if provenance:
        blob.setdefault("_provenance", {}).update(provenance)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=1))
    return out


def load_sequences(path) -> list[Record]:
    """Read a FASTA into Records, tolerating headers with no _IDR_x-y span.

    A header without a usable span becomes a record whose whole sequence is the IDR.

    Args:
        path (str | Path): FASTA file.

    Returns:
        list[Record]: One record per sequence, in file order.
    """
    out = []
    for header, seq in read_fasta(path):
        try:
            acc, start, end = parse_idr_header(header)
            if not 0 <= start < end <= len(seq):
                raise ValueError
        except ValueError:
            acc, start, end = header.split()[0], 0, len(seq)  # no span: the whole sequence is the IDR
        out.append(Record(acc, seq, start, end))
    return out


def length_match(positives, background, *, n, rng, bin_width=20):
    """Sample a background whose IDR-length distribution follows the positive set's.

    Length bins the background cannot fill are topped up from the rest of it, so the result is as
    close to n as the pool allows.

    Args:
        positives (list[Record]): Positive records.
        background (list[Record]): Candidate background records.
        n (int): Target background size.
        rng (np.random.Generator): Random source.
        bin_width (int): Length-bin width in residues.

    Returns:
        list[Record]: The sampled background.
    """
    def _bin(r):
        return (r.idr_end - r.idr_start) // bin_width

    pools: dict[int, list] = {}
    for r in background:
        pools.setdefault(_bin(r), []).append(r)

    pos_bins, counts = np.unique([_bin(r) for r in positives], return_counts=True)
    weights = counts / counts.sum()
    picked, shortfall = [], 0
    for b, w in zip(pos_bins.tolist(), weights.tolist()):
        want = int(round(w * n))
        pool = pools.get(b, [])
        take = min(want, len(pool))
        if take:
            idx = rng.choice(len(pool), size=take, replace=False)
            picked.extend(pool[i] for i in idx)
        shortfall += want - take
    if shortfall > 0:  # bins the background could not fill: top up from anywhere
        chosen = {id(r) for r in picked}
        rest = [r for r in background if id(r) not in chosen]
        if rest:
            idx = rng.choice(len(rest), size=min(shortfall, len(rest)), replace=False)
            picked.extend(rest[i] for i in idx)
    return picked
