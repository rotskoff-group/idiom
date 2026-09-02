"""Interpret and steer with a sparse autoencoder (IDiomSAE).

The SAE records its host model and layer, so from_pretrained loads the pair in one call.

    uv run python examples/03_sae_features.py --sae jxliu2/idiomsae-300M-L18-k32
"""

import argparse

import numpy as np

from idiom import IDiomSAE

SEQS = [
    "MEDSKVDNRPQACDEFGHIKLMNPQRSTVWY",
    "GSGSQPQPQPGSGSGSNNNNQQQQGSGSGS",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae", default="jxliu2/idiomsae-300M-L18-k32", help="HF repo id or local dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-steer", type=int, default=5)
    args = p.parse_args()

    sae = IDiomSAE.from_pretrained(args.sae, device=args.device)
    print(f"SAE on layer {sae.layer} of {sae.host_model} "
          f"(region={sae.region}, fim_mode={sae.fim_mode}, latents={sae.sae.num_latents})")

    # Feature activations, mean-pooled over each sequence's residues.
    feats, accessions = sae.encode(SEQS, pool="mean")
    print(f"features: {feats.shape}")
    for acc, row in zip(accessions, feats):
        top = np.argsort(row)[::-1][:5]
        print(f"  {acc}: top features {top.tolist()} (activations {row[top].round(3).tolist()})")

    # Causal steering: push generation along one feature's direction.
    feature = int(np.argsort(feats[0])[::-1][0])
    steered = sae.steer_generate(feature=feature, strength=0.5, n=args.n_steer, seed=0)
    print(f"steered on feature {feature}: {len(steered)} sequences")
    print("  example:", steered[0][:80])


if __name__ == "__main__":
    main()
