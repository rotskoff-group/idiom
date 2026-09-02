"""Find the SAE features enriched in your own sequences, then train toward them with RL-SAE.

This is the analysis half of the RL-SAE pipeline. Give it a set of sequences you care about (a
compartment, a functional class, hits from a screen) and it reports which SAE features fire in them
far more often than in a background, writes the top ones as a "signature", and plots the result.
The signature is directly consumable by the rl_sae reward, so the last thing this prints is
the idiom_grpo command that designs new sequences carrying that same feature code.

    python examples/05_feature_enrichment.py --positive my_seqs.fasta --name my_target --out enr/

The background defaults to the held-out validation split of the pretraining corpus, downloaded from
the Hub. A GPU is strongly recommended: the cost is dominated by encoding the background.

Input FASTAs may use IDiom's `_IDR_x-y` headers to mark the IDR inside each sequence; if the headers
have no span, each whole sequence is treated as the IDR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from idiom import IDiomSAE
from idiom.data.io import Record, read_fasta
from idiom.data.tokenizer import Tokenizer
from idiom.sae.features.enrichment import (
    FDR_ALPHA,
    LOG2OR_FLOOR,
    enrich,
    enriched_mask,
    feature_counts,
    top_features,
    write_signature,
)

_TOK = Tokenizer()
BACKGROUND_REPO = "jxliu2/idiom-data"
BACKGROUND_FILE = "training_sequences/validation.fasta"


def load_sequences(path) -> list[Record]:
    """Read a FASTA into Records, tolerating headers with no _IDR_x-y span.

    Args:
        path (str | Path): FASTA file.

    Returns:
        list[Record]: One record per sequence; without a span the whole sequence is the IDR.
    """
    from idiom.data.io import parse_idr_header

    out = []
    for header, seq in read_fasta(path):
        try:
            acc, start, end = parse_idr_header(header)
            if not 0 <= start < end <= len(seq):
                raise ValueError
        except ValueError:
            acc, start, end = header.split()[0], 0, len(seq)   # no span: whole sequence is the IDR
        out.append(Record(acc, seq, start, end))
    return out


def length_match(positives, background, *, n, rng, bin_width=20):
    """Sample a background whose IDR-length distribution follows the positive set's.

    Features that merely track length look enriched when the two sets have different length
    distributions, which they usually do. Matching removes most of that artifact.

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
    if shortfall > 0:      # bins the background could not fill: top up from anywhere
        chosen = {id(r) for r in picked}
        rest = [r for r in background if id(r) not in chosen]
        if rest:
            idx = rng.choice(len(rest), size=min(shortfall, len(rest)), replace=False)
            picked.extend(rest[i] for i in idx)
    return picked


def volcano(result, mask, out_path, name):
    """Plot |z| against log2 odds ratio, highlighting the enriched features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = result["active"]
    x, y = result["log2or"][active], np.abs(result["z"][active])
    enr = mask[active]
    fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
    ax.scatter(x[~enr], y[~enr], s=3, alpha=0.25, lw=0, color="#c3ced0", label="other", rasterized=True)
    ax.scatter(x[enr], y[enr], s=6, alpha=0.9, lw=0, color="#c1440e", label="enriched", rasterized=True)
    sig = result["fdr"][active] < FDR_ALPHA
    if sig.any():                       # the L-shaped enriched boundary
        zc = float(y[sig].min())
        ax.axvline(LOG2OR_FLOOR, ls="--", lw=0.8, color="#110d1b")
        ax.axhline(zc, ls="--", lw=0.8, color="#110d1b")
    ax.axvline(0, lw=0.8, color="#110d1b")
    ax.set_xlabel("log$_2$ odds ratio")
    ax.set_ylabel("|z|")
    ax.set_title(f"{name}: {int(mask.sum())} enriched features", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positive", required=True, help="FASTA of the sequences you care about")
    p.add_argument("--name", default="my_target", help="signature name (becomes sae_only_<name>)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--sae", default="jxliu2/idiomsae-300M-L18-k32", help="HF repo id or local dir")
    p.add_argument("--background", default=None,
                   help="background FASTA (default: the held-out validation split from the Hub)")
    p.add_argument("--max-background", type=int, default=10000,
                   help="background sequences to encode (dominates runtime)")
    p.add_argument("--top-n", type=int, default=30, help="features to keep in the signature")
    p.add_argument("--case", default="top30", help="case name to store the signature under")
    p.add_argument("--no-length-match", action="store_true",
                   help="skip length-matched background sampling (see the caveat in the docs)")
    p.add_argument("--no-drop-boundary", action="store_true",
                   help="keep features that only fire at IDR edges (excision artifacts)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sae = IDiomSAE.from_pretrained(args.sae, device=args.device)
    print(f"SAE: layer {sae.layer} of {sae.host_model}, {sae.sae.num_latents} latents")

    # --- sequences -----------------------------------------------------------------------
    positives = load_sequences(args.positive)
    print(f"positive set: {len(positives)} sequences from {args.positive}")

    if args.background:
        bg_path = args.background
    else:
        from huggingface_hub import hf_hub_download
        print(f"background: downloading {BACKGROUND_FILE} from {BACKGROUND_REPO}")
        bg_path = hf_hub_download(BACKGROUND_REPO, BACKGROUND_FILE, repo_type="dataset")
    background = load_sequences(bg_path)
    print(f"background pool: {len(background)} sequences")

    if args.no_length_match:
        idx = rng.choice(len(background), size=min(args.max_background, len(background)), replace=False)
        background = [background[i] for i in idx]
        print(f"background: {len(background)} sampled uniformly (NOT length-matched)")
    else:
        background = length_match(positives, background, n=args.max_background, rng=rng)
        print(f"background: {len(background)} sampled length-matched to the positive set")

    # --- encode both sets through the SAE ------------------------------------------------
    print("encoding positives ...")
    pos_fd = sae.build_feature_dataset(positives, out / "fd_positive", batch_size=args.batch_size)
    print("encoding background ... (this is the slow part)")
    bg_fd = sae.build_feature_dataset(background, out / "fd_background", batch_size=args.batch_size)

    # --- enrichment -----------------------------------------------------------------------
    a, n_pos = feature_counts(pos_fd)
    b, n_neg = feature_counts(bg_fd)
    result = enrich(a, n_pos, b, n_neg, sae.sae.num_latents)
    mask = enriched_mask(result)
    print(f"\n{int(mask.sum())} enriched features "
          f"(FDR < {FDR_ALPHA}, log2OR >= {LOG2OR_FLOOR}, prevalence >= 5%)")

    tsv = out / "enrichment.tsv"
    order = np.argsort(-result["log2or"])
    with tsv.open("w") as fh:
        fh.write("feature\ta_fire\tprev_pos\tb_fire\tprev_neg\tlog2OR\tz\tp\tfdr\tenriched\n")
        for f in order:
            if not result["active"][f]:
                continue
            fh.write(f"{f}\t{int(result['a'][f])}\t{result['prev_pos'][f]:.4f}\t"
                     f"{int(result['b'][f])}\t{result['prev_neg'][f]:.4f}\t{result['log2or'][f]:.3f}\t"
                     f"{result['z'][f]:.2f}\t{result['p'][f]:.2e}\t{result['fdr'][f]:.2e}\t"
                     f"{int(mask[f])}\n")
    print(f"  wrote {tsv}")

    # --- signature ------------------------------------------------------------------------
    ids = top_features(result, n=args.top_n, drop_boundary=not args.no_drop_boundary,
                       feature_dir=bg_fd)
    sig_path = write_signature(
        out / "signature.json", {args.name: ids}, case=args.case,
        provenance={"sae": args.sae, "positive": str(args.positive), "background": str(bg_path),
                    "n_pos": int(n_pos), "n_background": int(n_neg),
                    "length_matched": not args.no_length_match,
                    "boundary_dropped": not args.no_drop_boundary,
                    "rank": "log2 odds ratio, descending"})
    print(f"  wrote {sig_path}  ({len(ids)} features: {ids[:8]}{' ...' if len(ids) > 8 else ''})")

    png = out / "enrichment.png"
    volcano(result, mask, png, args.name)
    print(f"  wrote {png}")

    # --- what to do with it ---------------------------------------------------------------
    print(f"\nTrain a model to reproduce this feature code:\n"
          f"  IDIOM_SAEREWARD_FEATURES={sig_path} IDIOM_SAEREWARD_CASE={args.case} \\\n"
          f"    idiom_grpo init_from=/path/to/base.ckpt \\\n"
          f"      reward.rl_sae.enabled=true reward.rl_sae.signature={args.name}")


if __name__ == "__main__":
    main()
