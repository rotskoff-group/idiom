"""``python -m eval.run`` — fast-profile evaluation of an IDiom checkpoint.

Generates (or loads) sequences and reports: generation validity, sparrow feature summaries,
Wasserstein-1 vs a reference set (e.g. DisProt IDRs), and held-out perplexity. Writes
``metrics.json`` + a per-sequence feature CSV; optionally logs to wandb.

    python -m eval.run --ckpt last.ckpt --n-layers 24 --d-model 1024 --n-heads 16 \
        --generate-n 10000 --reference disprot_idrs.fasta --test-fasta test.fasta --out eval_out

Custom arch via --n-layers/--d-model/--n-heads, or a named --size (12l/24l/36l). The heavier
novelty (mmseqs) / disorder (IUPred/ColabFold) metrics are a separate full profile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    from idiom.api import IDiom
    from idiom.data.io import read_fasta
    from eval.distances import w1_table
    from eval.metrics import FEATURES, features_table, summarize
    from eval.validity import validity_stats
    from idiom.model import idiom_12l, idiom_24l, idiom_36l
    from idiom.model.config import ModelConfig

    sizes = {"12l": idiom_12l, "24l": idiom_24l, "36l": idiom_36l}
    p = argparse.ArgumentParser(description="Fast-profile evaluation of an IDiom checkpoint.")
    p.add_argument("--ckpt", required=True, help="lightning .ckpt")
    p.add_argument("--size", choices=list(sizes), default="24l", help="named arch")
    p.add_argument("--n-layers", type=int)
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--generations", help="FASTA of pre-made sequences (skip generation)")
    p.add_argument("--generate-n", type=int, default=10000, help="de-novo IDPs to generate if no --generations")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reference", help="FASTA of reference IDRs (e.g. DisProt) for W1")
    p.add_argument("--test-fasta", help="record FASTA for held-out perplexity")
    p.add_argument("--ppl-max-records", type=int, default=20000)
    # full profile (opt-in):
    p.add_argument("--disorder", action="store_true", help="metapredict disorder of generations (+ reference)")
    p.add_argument("--iupred", action="store_true", help="IUPred3 disorder (orthogonal; external tool)")
    p.add_argument("--iupred-path", help="extracted iupred3/ dir (default: $IUPRED3_PATH / known location)")
    p.add_argument("--novelty-ref", help="training-IDR FASTA; mmseqs max-identity per generation (memorization)")
    p.add_argument("--mmseqs", help="mmseqs binary path (default: PATH / known location)")
    p.add_argument("--out", required=True)
    p.add_argument("--wandb-project", help="if set, log metrics to this wandb project")
    args = p.parse_args()

    if args.n_layers and args.d_model and args.n_heads:
        cfg = ModelConfig(n_layers=args.n_layers, d_model=args.d_model,
                          n_heads=args.n_heads, max_seq_len=args.max_seq_len)
    else:
        cfg = sizes[args.size]()
    idiom = IDiom.from_lightning_checkpoint(args.ckpt, cfg, device="auto")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. generations (load or sample)
    if args.generations:
        gens = [s for _, s in read_fasta(args.generations)]
    else:
        gens = idiom.generate_idp(n=args.generate_n, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature)
        with (out / "generated_idp.fasta").open("w") as f:
            for i, s in enumerate(gens):
                f.write(f">idiom_idp_{i}\n{s}\n")

    results: dict = {"n_generated": len(gens)}
    results["validity"] = validity_stats(gens, args.max_new_tokens)

    ref_seqs = [s for _, s in read_fasta(args.reference)] if args.reference else None

    # 2. sparrow features + W1 vs reference
    gen_tab = features_table(gens)
    results["generated_summary"] = summarize(gen_tab)
    if ref_seqs is not None:
        ref_tab = features_table(ref_seqs)
        results["reference_summary"] = summarize(ref_tab)
        results["w1_vs_reference"] = w1_table(gen_tab, ref_tab)

    # 2b. disorder (metapredict + optional IUPred3) — full profile
    if args.disorder or args.iupred:
        from eval.disorder import disorder_stats, iupred3_disorder, metapredict_disorder
        if args.disorder:
            results["disorder_generated"] = disorder_stats(metapredict_disorder(gens, device=str(idiom.device)))
            if ref_seqs is not None:
                results["disorder_reference"] = disorder_stats(metapredict_disorder(ref_seqs, device=str(idiom.device)))
        if args.iupred:
            results["iupred_generated"] = disorder_stats(iupred3_disorder(gens, iupred_path=args.iupred_path))
            if ref_seqs is not None:
                results["iupred_reference"] = disorder_stats(iupred3_disorder(ref_seqs, iupred_path=args.iupred_path))

    # 2c. novelty / memorization (mmseqs vs training corpus) — full profile
    if args.novelty_ref:
        from eval.novelty import max_identity, novelty_stats
        results["novelty"] = novelty_stats(max_identity(gens, args.novelty_ref, mmseqs=args.mmseqs))

    # 3. held-out perplexity
    if args.test_fasta:
        from eval.perplexity import perplexity
        results["perplexity"] = perplexity(
            idiom.model, args.test_fasta, max_len=cfg.max_seq_len,
            device=idiom.device, max_records=args.ppl_max_records)

    # 4. write outputs
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    with (out / "generated_features.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FEATURES)
        for i in range(len(gens)):
            w.writerow([gen_tab[k][i] for k in FEATURES])

    print(f"\n== eval ({len(gens)} generated) ==")
    v = results["validity"]
    print(f"  validity: {100*v['frac_terminated']:.0f}% terminated, "
          f"len median {v['length_median']:.0f}, {100*v['frac_unique']:.0f}% unique")
    if "perplexity" in results:
        print(f"  perplexity: {results['perplexity']['perplexity']:.3f} "
              f"(nll {results['perplexity']['nll']:.3f}, {results['perplexity']['n_tokens']:,} tokens)")
    if "disorder_generated" in results:
        dg = results["disorder_generated"]
        ref = f" (ref {results['disorder_reference']['mean_disorder']:.3f})" if "disorder_reference" in results else ""
        print(f"  disorder: gen mean {dg['mean_disorder']:.3f}{ref}")
    if "novelty" in results:
        nv = results["novelty"]
        print(f"  novelty: max-id mean {nv['max_identity_mean']:.3f}, frac>=0.9 {nv['frac_ge_0.9']:.3f}")
    if "w1_vs_reference" in results:
        print("  W1 vs reference (normalized):")
        for feat, d in results["w1_vs_reference"].items():
            print(f"    {feat:18s} {d['w1_norm']:.3f}")
    print(f"  -> {out/'metrics.json'}")

    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, dir=str(out))
        flat = {f"w1/{k}": v["w1_norm"] for k, v in results.get("w1_vs_reference", {}).items()}
        flat.update({f"validity/{k}": v for k, v in results["validity"].items()})
        if "perplexity" in results:
            flat["perplexity"] = results["perplexity"]["perplexity"]
        if "disorder_generated" in results:
            flat.update({f"disorder/{k}": v for k, v in results["disorder_generated"].items()})
        if "novelty" in results:
            flat.update({f"novelty/{k}": v for k, v in results["novelty"].items()})
        run.log(flat)
        run.finish()


if __name__ == "__main__":
    main()
