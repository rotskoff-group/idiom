"""Read and steer SAE features.

A sparse autoencoder trained on one of IDiom's layers decomposes the residual stream into a large
set of features, most of them off at any given residue. This script reads which features fire on a
sequence, then steers generation along one of them -- pushing the model's own representation and
watching the sequences change.

    uv run cookbook/scripts/python/sae_features.py

The SAE records its host model and layer, so from_pretrained loads the pair in one call. A GPU is
recommended; DEVICE = "auto" falls back to CPU.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from idiom import IDiomSAE

matplotlib.use("Agg")  # the figure is written to a file, not displayed

SAE = "jxliu2/idiomsae-300M-L18-k32"   # HF repo id or a local directory
DEVICE = "auto"
N_STEER = 60                           # sequences per steering strength
STRENGTHS = [0.0, 0.5, 1.0]            # 0.0 is the unsteered baseline
FIGURE = "steering_composition.png"    # where the composition plot is written

sae = IDiomSAE.from_pretrained(SAE, device=DEVICE)
print(f"SAE on layer {sae.layer} of {sae.host_model}")
print(f"region={sae.region}  fim_mode={sae.fim_mode}  latents={sae.sae.num_latents}")


# Which features fire
# -------------------
# encode returns one activation row per sequence when pooled, and the top entries of that row are
# the features that describe it.

SEQS = [
    "MEDSKVDNRPQACDEFGHIKLMNPQRSTVWY",
    "GSGSQPQPQPGSGSGSNNNNQQQQGSGSGS",
]

feats, accessions = sae.encode(SEQS, pool="mean")
print(f"\nfeatures: {feats.shape}\n")

for acc, row in zip(accessions, feats):
    top = np.argsort(row)[::-1][:5]
    print(f"{acc}")
    for f in top:
        print(f"    feature {f:>5}   activation {row[f]:.3f}")


# Steering
# --------
# steer_generate adds a feature's decoder direction to the residual stream while generating, so the
# model produces sequences carrying whatever that feature encodes. Strength 0 runs the same code
# path without the push, which makes it the honest baseline to compare against.

feature = int(np.argsort(feats[0])[::-1][0])
print(f"\nsteering feature {feature}\n")

runs = {}
for s in STRENGTHS:
    runs[s] = sae.steer_generate(feature=feature, strength=s, n=N_STEER, seed=0)
    lengths = [len(x) for x in runs[s]]
    print(f"strength {s:>4}:  {len(runs[s])} sequences, mean length {np.mean(lengths):.0f}")
    print(f"              {runs[s][0][:70]}")

# The clearest read on what changed is composition. Plotting each steered run against the unsteered
# baseline shows which residues the feature pushes for and against.

AA = "ACDEFGHIKLMNPQRSTVWY"


def composition(seqs):
    total = sum(len(s) for s in seqs)
    return np.array([sum(s.count(a) for s in seqs) / total for a in AA])


base = composition(runs[0.0])
steered = [s for s in STRENGTHS if s != 0.0]

fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
width = 0.8 / len(steered)
for i, s in enumerate(steered):
    delta = composition(runs[s]) - base
    ax.bar(np.arange(len(AA)) + i * width, delta, width, label=f"strength {s}")
ax.axhline(0, lw=0.8, color="#110d1b")
ax.set_xticks(np.arange(len(AA)) + width * (len(steered) - 1) / 2)
ax.set_xticklabels(list(AA))
ax.set_ylabel("frequency vs. unsteered")
ax.set_title(f"composition shift under feature {feature}", fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIGURE, dpi=200)
print(f"\nwrote {FIGURE}")


# Next
# ----
# A feature that moves composition sharply is one worth naming. To find out which features matter
# for a set of sequences you care about -- rather than picking the most active one on an arbitrary
# example -- run feature_enrichment.py, which finds the features enriched in your own sequences and
# writes the signature that trains a model to reproduce them.
