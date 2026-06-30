"""``python -m eval.run_sae_eval`` — held-out eval of a layer-swept SAE set.

Repo-only operator/repro code (run from the repo root): it imports the shipped library's metric
primitives (:func:`idiom.sae.eval.compute_fidelity` via ``IDiomSAE.fidelity`` and
:func:`idiom.sae.eval.reconstruction.reconstruction_stats`), sweeps a directory of SAE releases, and
writes the per-layer CSV the SAE figures consume. It is not part of the installed ``idiom`` wheel.

For each released SAE dir (``sae_config.json`` + ``sae.safetensors``) under ``--sae-dir`` matching
``--glob``, computes the metrics the SAE figures consume, on a held-out record FASTA, on the
distribution the SAE was trained on (its recorded ``region`` + ``fim_mode``):

  - **downstream fidelity** (Gao "loss recovered"): clean / SAE / ablate NLL + ``pct_loss_recovered``
    via :func:`idiom.sae.eval.fidelity.compute_fidelity` (``IDiomSAE.fidelity``).
  - **reconstruction**: FVU + explained variance over the held-out activations.
  - **sparsity**: mean L0 (active latents/token; ~k for top-k) and dead-feature fraction.
  - **feature density**: per-latent activation frequency (saved per layer for the density histogram).

The host model is loaded once and shared across the SAEs (all SAEs under one ``--sae-dir`` share a
host model). Point ``--sae-dir`` at a *single variant* (e.g. ``.../03_sae/x16_k32_idp``); the
per-layer outputs are keyed by layer, so evaluate variants into separate ``--out`` dirs.

``--fim-mode``:
  - ``auto`` (default) — each SAE is evaluated in its own recorded ``fim_mode`` (idp -> de-novo,
    ``fim_idr_prob=0``; idr -> flank-conditioned, ``fim_idr_prob=1``). This keeps every SAE
    on-distribution and is the right choice for a mixed set.
  - ``idp`` / ``idr`` — force the eval prompt format for all SAEs regardless of how they were
    trained (off-distribution if it disagrees with the SAE's ``fim_mode``; for special studies).

    python -m eval.run_sae_eval --sae-dir DIR/x16_k32_idp --glob 'L*' \
        --val FASTA --out DIR/eval/x16_k32_idp --max-records 2000 --fim-mode auto --device auto
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

# fim_mode -> fim_idr_prob used at eval time (the SAE's training prompt format).
_FIM_IDR_PROB = {"idp": 0.0, "idr": 1.0}


def _eval_fim_idr_prob(sae_fim_mode: str, override: str) -> tuple[str, float]:
    """Resolve (effective fim_mode, fim_idr_prob) for one SAE given the --fim-mode flag."""
    mode = sae_fim_mode if override == "auto" else override
    if mode not in _FIM_IDR_PROB:
        raise SystemExit(f"unknown fim_mode {mode!r} (expected idp|idr)")
    return mode, _FIM_IDR_PROB[mode]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sae-dir", required=True, help="parent dir of the per-layer SAE release dirs")
    ap.add_argument("--glob", default="L*", help="glob (under --sae-dir) for the SAE dirs")
    ap.add_argument("--val", required=True, help="held-out record FASTA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-records", type=int, default=2000,
                    help="random held-out subset size (the NLL ratio converges well before full val)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random held-out subsample")
    ap.add_argument("--fidelity-batch", type=int, default=16)
    ap.add_argument("--fim-mode", choices=["auto", "idp", "idr"], default="auto",
                    help="eval prompt format: auto = per-SAE recorded fim_mode (on-distribution)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from idiom import IDiom, IDiomSAE
    from idiom.sae.eval.reconstruction import reconstruction_stats
    from idiom.utils.device import resolve_device

    dev = resolve_device(args.device)
    out = Path(args.out)
    (out / "scores" / "feature_freq").mkdir(parents=True, exist_ok=True)

    sae_dirs = sorted(d for d in glob.glob(str(Path(args.sae_dir) / args.glob)) if Path(d).is_dir())
    if not sae_dirs:
        raise SystemExit(f"no SAE dirs under {args.sae_dir}/{args.glob}")

    # Subset the held-out set once: IDiomSAE.fidelity() reads the WHOLE FASTA, so cap up front.
    # Reservoir sampling -> a uniform RANDOM sample over the full val split (file order can carry
    # structure), deterministic given --seed, single streaming pass (never holds the whole split).
    import random

    from idiom.data.io import read_records
    rng = random.Random(args.seed)
    reservoir = []
    n_seen = 0
    for r in read_records(args.val):
        n_seen += 1
        if len(reservoir) < args.max_records:
            reservoir.append(r)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < args.max_records:
                reservoir[j] = r
    val = out / "_val_subset.fasta"
    with val.open("w") as fh:
        for r in reservoir:
            fh.write(f">{r.accession}_IDR_{r.idr_start + 1}-{r.idr_end}\n{r.full_seq}\n")
    nv = len(reservoir)
    print(f"held-out subset: {nv} random of {n_seen} records (seed={args.seed}) -> {val}  "
          f"(fim-mode={args.fim_mode})", flush=True)

    host = None
    rows = []
    for d in sae_dirs:
        cfg = json.loads((Path(d) / "sae_config.json").read_text())
        if host is None:
            host = IDiom.load(cfg["host_model"], device=dev)
        sae = IDiomSAE.from_pretrained(d, model=host, device=dev)
        layer, region = sae.layer, sae.region
        eff_mode, ffp = _eval_fim_idr_prob(sae.fim_mode, args.fim_mode)
        print(f"[sae L{layer}] region={region} sae_fim_mode={sae.fim_mode} "
              f"eval_fim_mode={eff_mode} fim_idr_prob={ffp}", flush=True)

        fid = sae.fidelity(str(val), batch_size=args.fidelity_batch, fim_idr_prob=ffp)
        rec = reconstruction_stats(host, sae, layer, region, ffp, str(val), device=dev,
                                   max_records=args.max_records)
        np.save(out / "scores" / "feature_freq" / f"L{layer}.npy", rec.feature_freq)
        rows.append({
            "layer": layer, "region": region, "fim_mode": eff_mode,
            "loss_clean": fid.loss_clean, "loss_sae": fid.loss_sae, "loss_ablate": fid.loss_ablate,
            "pct_loss_recovered": fid.pct_loss_recovered,
            "fvu": rec.fvu, "explained_var": rec.explained_var, "l0_mean": rec.l0_mean,
            "frac_dead": rec.frac_dead, "n_dead": rec.n_dead, "num_latents": rec.num_latents,
            "n_active_rows": rec.n_active_rows,
        })
        print(f"  fvu={rec.fvu:.4f} EV={rec.explained_var:.4f} l0={rec.l0_mean:.1f} "
              f"frac_dead={rec.frac_dead:.3f} pct_recovered={fid.pct_loss_recovered:.1f}", flush=True)

    rows.sort(key=lambda r: r["layer"])
    with (out / "scores" / "fidelity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(
        {"sae_dir": args.sae_dir, "val": args.val, "n_records": nv,
         "fim_mode": args.fim_mode, "layers": rows}, indent=2, default=float))
    print(f"-> {out / 'scores' / 'fidelity.csv'}", flush=True)


if __name__ == "__main__":
    main()
