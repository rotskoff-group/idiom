"""Draw sequence logos of what SAE features detect — the grammar behind an enrichment signature.

Pairs with 05_feature_enrichment.py: that script finds the features enriched in a set and writes a
signature; this one shows, for each of those features, the residue grammar it fires on. For a
feature it takes the top-activating windows across the positive set, stacks them, and renders an
information-content logo (logomaker). Point --signature at 05's signature.json to logo exactly the
enriched features, or omit it to logo the most active features on the set.

    uv run python examples/python/06_feature_logos.py --positive examples/example_data/protgps/nucleolus.fasta

A GPU is recommended (the cost is encoding the positive set through the SAE); pass --device cpu to
force CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from idiom import IDiomSAE


def per_sequence_activations(feats, index):
    """Group per-residue SAE activations back into per-accession (residues, row_indices).

    encode(pool="none") returns residue rows in order; this regroups them so a window can be cut
    from a single sequence's residue string.

    Args:
        feats (np.ndarray): [N_res, num_latents] per-residue activations.
        index (list[dict]): Per-row metadata with accession/source_pos/residue.

    Returns:
        list[tuple[str, np.ndarray]]: One (residue_string, row_indices) per accession, in order.
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


def top_windows(feature_id, feats, per_seq, index, *, n_windows, half_width):
    """Return the top-activating fixed-width residue windows for one feature.

    Args:
        feature_id (int): The SAE latent to profile.
        feats (np.ndarray): [N_res, num_latents] activations.
        per_seq (list): Output of per_sequence_activations.
        index (list[dict]): Per-row metadata (unused directly; kept for symmetry).
        n_windows (int): Number of windows (one per top sequence).
        half_width (int): Residues on each side of the peak; window length is 2*half_width+1.

    Returns:
        list[str]: Equal-length residue windows, most-active first.
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


def pick_features(args, feats, num_latents):
    """Choose which features to logo: the enrichment signature if given, else the most active.

    Args:
        args: Parsed CLI args (signature/case/name/n_features).
        feats (np.ndarray): [N_res, num_latents] activations over the positive set.
        num_latents (int): SAE latent count (for bounds only).

    Returns:
        list[int]: Feature ids to render.
    """
    if args.signature:
        blob = json.loads(Path(args.signature).read_text())
        case = args.case or next(k for k in blob if not k.startswith("_"))
        sets = blob[case]
        name = args.name or next(iter(sets))
        ids = list(sets[name])
        print(f"signature: {len(ids)} features for {name!r} (case {case!r}); showing top {args.n_features}")
        return ids[:args.n_features]
    mean_act = feats.mean(0)
    ids = [int(f) for f in np.argsort(-mean_act) if mean_act[f] > 0][:args.n_features]
    print(f"no signature given; showing the {len(ids)} most active features on the positive set")
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positive", default="examples/example_data/protgps/nucleolus.fasta",
                   help="FASTA whose feature grammar to logo")
    p.add_argument("--sae", default="jxliu2/idiomsae-300M-L18-k32", help="HF repo id or local dir")
    p.add_argument("--signature", default=None, help="signature.json from 05 (else top active feats)")
    p.add_argument("--case", default=None, help="case in the signature file (default: first)")
    p.add_argument("--name", default=None, help="set name in the signature (default: first)")
    p.add_argument("--n-features", type=int, default=6, help="features to logo")
    p.add_argument("--n-windows", type=int, default=60, help="top windows stacked per feature")
    p.add_argument("--half-width", type=int, default=7, help="residues each side of the peak")
    p.add_argument("--out", default="feature_logos.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import logomaker
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sae = IDiomSAE.from_pretrained(args.sae, device=args.device)
    print(f"SAE: layer {sae.layer} of {sae.host_model}, {sae.sae.num_latents} latents")

    print(f"encoding {args.positive} ...")
    feats, index = sae.encode(args.positive, pool="none")
    per_seq = per_sequence_activations(feats, index)
    print(f"  {feats.shape[0]} residues across {len(per_seq)} sequences")

    features = pick_features(args, feats, sae.sae.num_latents)

    fig, axes = plt.subplots(len(features), 1, figsize=(6, 1.5 * len(features)),
                             constrained_layout=True, squeeze=False)
    for ax, fid in zip(axes[:, 0], features):
        windows = top_windows(fid, feats, per_seq, index,
                              n_windows=args.n_windows, half_width=args.half_width)
        if len(windows) < 2:
            ax.set_axis_off()
            ax.set_title(f"feature {fid}: too few windows", fontsize=8)
            continue
        mat = logomaker.alignment_to_matrix(windows, to_type="information")
        logomaker.Logo(mat, ax=ax, color_scheme="chemistry")
        ax.set_title(f"feature {fid}  ({len(windows)} windows)", fontsize=8)
        ax.set_ylabel("bits", fontsize=7)
        ax.set_xticks([])
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}  ({len(features)} feature logos)")


if __name__ == "__main__":
    main()
