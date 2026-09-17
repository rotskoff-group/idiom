"""CLI for SAE feature enrichment against a length-matched background."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from idiom.sae.features.enrichment import (
    FDR_ALPHA,
    LOG2OR_FLOOR,
    MIN_TOTAL_FIRE,
    PREV_POS_FLOOR,
    SMOOTH,
    boundary_features,
    enrich,
    enriched_mask,
    feature_counts,
    length_match,
    load_sequences,
    write_signature,
)

DATA_REPO = "jxliu2/idiom-db"
VALIDATION_FASTA = "idiom-db/idiom-db-v1_validation.fasta"


def _positive_int(value: str) -> int:
    """Parse a positive integer or raise an argparse validation error."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> None:
    """Download or load inputs, encode IDRs, and export enrichment and a signature."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae", required=True, help="SAE Hub ID or release directory")
    p.add_argument("--positive", required=True, type=Path, help="positive IDR FASTA")
    p.add_argument("--background", type=Path, help="local FASTA; defaults to the HF validation split")
    p.add_argument("--out", required=True, type=Path, help="new or empty output directory")
    p.add_argument("--name", required=True, help="signature name for the GRPO reward")
    p.add_argument("--case", help="signature case (default: top<TOP_N>)")
    p.add_argument("--top-n", type=_positive_int, default=30)
    p.add_argument("--max-positive", type=_positive_int, help="sample at most this many positives")
    p.add_argument(
        "--max-background", type=_positive_int, default=10000, help="target background sample size"
    )
    p.add_argument("--batch-size", type=_positive_int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA)
    p.add_argument("--log2or-floor", type=float, default=LOG2OR_FLOOR)
    p.add_argument("--prev-pos-floor", type=float, default=PREV_POS_FLOOR)
    p.add_argument("--min-total-fire", type=_positive_int, default=MIN_TOTAL_FIRE)
    p.add_argument("--keep-boundary", action="store_true", help="disable boundary-feature filtering")
    args = p.parse_args(argv)
    if not 0 < args.fdr_alpha <= 1 or not 0 <= args.prev_pos_floor <= 1:
        p.error("--fdr-alpha must be in (0, 1] and --prev-pos-floor in [0, 1]")
    if not math.isfinite(args.log2or_floor) or args.seed < 0:
        p.error("--log2or-floor must be finite and --seed nonnegative")
    if not args.name.strip() or (args.case is not None and not args.case.strip()):
        p.error("--name and --case must not be empty")
    for path in (args.positive, args.background):
        if path is not None and not path.is_file():
            p.error(f"FASTA does not exist: {path}")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error("--out must be a new or empty directory to avoid stale results")
    args.case = args.case or f"top{args.top_n}"

    from idiom import IDiomSAE

    sae = IDiomSAE.from_pretrained(args.sae, device=args.device)
    if sae.fim_mode != "unprompted" or sae.region != "idr":
        p.error("enrichment requires an SAE trained on unprompted IDRs")
    max_length = sae.model.cfg.max_seq_len - 4

    def usable(path):
        """Load nonempty IDRs that fit the context limit and report the retained count."""
        records = load_sequences(path)
        kept = [r for r in records if 0 < r.idr_end - r.idr_start <= max_length]
        print(f"{path}: kept {len(kept)} of {len(records)} canonical records within context")
        return kept

    positives = usable(args.positive)
    if args.max_positive is not None and len(positives) > args.max_positive:
        chosen = np.random.default_rng(args.seed).choice(len(positives), args.max_positive, replace=False)
        positives = [positives[i] for i in sorted(chosen)]
    if not positives:
        p.error("no usable positive records")
    background_path = args.background
    if background_path is None:
        from huggingface_hub import hf_hub_download

        background_path = Path(hf_hub_download(DATA_REPO, VALIDATION_FASTA, repo_type="dataset"))
    positive_idrs = {r.full_seq[r.idr_start : r.idr_end] for r in positives}
    pool = [r for r in usable(background_path) if r.full_seq[r.idr_start : r.idr_end] not in positive_idrs]
    background = length_match(positives, pool, n=args.max_background, rng=np.random.default_rng(args.seed))
    if not background:
        p.error("no usable background records remain after excluding exact positive IDR matches")
    print(f"Encoding {len(positives)} positives and {len(background)} background records (pool: {len(pool)})")
    args.out.mkdir(parents=True, exist_ok=True)
    pos_fd = sae.build_feature_dataset(positives, args.out / "fd_positive", batch_size=args.batch_size)
    bg_fd = sae.build_feature_dataset(background, args.out / "fd_background", batch_size=args.batch_size)
    a, n_pos = feature_counts(pos_fd)
    b, n_neg = feature_counts(bg_fd)
    result = enrich(a, n_pos, b, n_neg, sae.sae.num_latents, min_total_fire=args.min_total_fire)
    mask = enriched_mask(
        result, fdr_alpha=args.fdr_alpha, log2or_floor=args.log2or_floor, prev_pos_floor=args.prev_pos_floor
    )
    candidates = np.flatnonzero(mask)
    bad = set() if args.keep_boundary else boundary_features(bg_fd, candidates)
    ranked = candidates[np.argsort(-result["log2or"][candidates], kind="stable")]
    ids = [int(f) for f in ranked if f not in bad][: args.top_n]
    columns = ["a", "b", "prev_pos", "prev_neg", "log2or", "z", "p", "fdr", "active"]
    with (args.out / "enrichment.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["feature_id", *columns, "enriched", "boundary_filtered", "selected"])
        for f in range(sae.sae.num_latents):
            writer.writerow([f, *(result[key][f] for key in columns), bool(mask[f]), f in bad, f in ids])
    provenance = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "background": str(background_path),
        "background_repo": DATA_REPO if args.background is None else None,
        "background_file": VALIDATION_FASTA if args.background is None else None,
        "host_model": str(sae.host_model),
        "layer": sae.layer,
        "n_pos": n_pos,
        "n_background": n_neg,
        "background_pool_size": len(pool),
        "length_matched": True,
        "length_bin_width": 20,
        "smooth": SMOOTH,
        "exact_positive_idrs_excluded": True,
        "boundary_dropped": not args.keep_boundary,
        "rank": "log2 odds ratio, descending",
        "selected_features": ids,
    }
    (args.out / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if ids:
        write_signature(args.out / "signature.json", {args.name: ids}, case=args.case, provenance=provenance)
        print(f"Saved {len(ids)} signature features; SIGNATURE={args.name}, CASE={args.case}")
    else:
        print("No features passed the filters; no signature was written.")
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
