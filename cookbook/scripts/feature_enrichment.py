"""Feature enrichment: from a set of sequences to an RL target.

Give this script a set of sequences you care about -- a compartment, a functional class, hits from
a screen -- and it finds which SAE features fire in them far more often than in a background,
writes the strongest as a signature, and shows the residue grammar behind each one.

    uv run cookbook/scripts/feature_enrichment.py

The signature is directly consumable by the sae_only_<name> reward, so the end of this script is
the command that designs new sequences carrying the same feature code.

A GPU is strongly recommended: the cost is dominated by encoding the background.
"""

from pathlib import Path

import logomaker
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download

from idiom import IDiomSAE
from idiom.sae.features import per_sequence_activations, top_windows
from idiom.sae.features.enrichment import (
    FDR_ALPHA, LOG2OR_FLOOR, enrich, enriched_mask, feature_counts,
    length_match, load_sequences, top_features, write_signature,
)

matplotlib.use("Agg")  # the figures are written to files, not displayed

REPO = Path(__file__).resolve().parents[2]  # so the script runs from any directory

POSITIVE = str(REPO / "cookbook/example_data/protgps/nucleolus.fasta")  # the set you care about
NAME = "nucleolus"          # signature name; the reward becomes sae_only_<name>
SAE = "jxliu2/idiomsae-300M-L18-k32"
DEVICE = "auto"
OUT = "enr"                 # working directory for the feature datasets and signature
MAX_BACKGROUND = 10000      # background sequences to encode -- this dominates runtime
TOP_N = 30                  # features kept in the signature
CASE = "top30"              # case name to store the signature under
SEED = 0
N_FEATURES = 6              # features to logo
N_WINDOWS = 60              # top windows stacked per feature
HALF_WIDTH = 7              # residues each side of the peak

out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

sae = IDiomSAE.from_pretrained(SAE, device=DEVICE)
print(f"SAE: layer {sae.layer} of {sae.host_model}, {sae.sae.num_latents} latents")


# The two sets
# ------------
# load_sequences reads a FASTA whether or not its headers carry an IDiom _IDR_x-y span; without
# one, the whole sequence is treated as the IDR.

positives = load_sequences(POSITIVE)
print(f"positive set: {len(positives)} sequences from {POSITIVE}")

# The background defaults to the held-out validation split of the pretraining corpus.
#
# It is sampled length-matched to the positive set: features that merely track length look enriched
# when the two sets have different length distributions, which they usually do.

bg_path = hf_hub_download("jxliu2/idiom-data", "training_sequences/validation.fasta",
                          repo_type="dataset")
background = load_sequences(bg_path)
print(f"background pool: {len(background)} sequences")

background = length_match(positives, background, n=MAX_BACKGROUND, rng=rng)
print(f"sampled: {len(background)} length-matched to the positive set")


# Encode both sets through the SAE
# --------------------------------
# This is the slow part -- the background is the bulk of it.

pos_fd = sae.build_feature_dataset(positives, out / "fd_positive", batch_size=16)
bg_fd = sae.build_feature_dataset(background, out / "fd_background", batch_size=16)
print("done")


# Enrichment
# ----------
# A feature "fires" in a sequence if the SAE selects it at any residue, counted once per sequence.
# From the two firing counts, enrich computes a Haldane-Anscombe log2 odds ratio, standardizes it
# against a hypergeometric null, and controls the false discovery rate with Benjamini-Hochberg.

a, n_pos = feature_counts(pos_fd)
b, n_neg = feature_counts(bg_fd)
result = enrich(a, n_pos, b, n_neg, sae.sae.num_latents)
mask = enriched_mask(result)

print(f"{int(mask.sum())} enriched features")
print(f"  FDR < {FDR_ALPHA}, log2 odds ratio >= {LOG2OR_FLOOR}, prevalence >= 5%")

active = result["active"]
x, y = result["log2or"][active], np.abs(result["z"][active])
enr = mask[active]

fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
ax.scatter(x[~enr], y[~enr], s=4, alpha=0.25, lw=0, color="#c3ced0", label="other")
ax.scatter(x[enr], y[enr], s=8, alpha=0.9, lw=0, color="#c1440e", label="enriched")
sig = result["fdr"][active] < FDR_ALPHA
if sig.any():                                  # the L-shaped enriched boundary
    ax.axhline(float(y[sig].min()), ls="--", lw=0.8, color="#110d1b")
    ax.axvline(LOG2OR_FLOOR, ls="--", lw=0.8, color="#110d1b")
ax.axvline(0, lw=0.8, color="#110d1b")
ax.set_xlabel("log$_2$ odds ratio")
ax.set_ylabel("|z|")
ax.set_title(f"{NAME}: {int(mask.sum())} enriched features", fontsize=10)
ax.legend(loc="upper left", fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(out / "volcano.png", dpi=200)
print(f"wrote {out / 'volcano.png'}")


# The signature
# -------------
# top_features keeps the strongest by odds ratio and drops boundary features -- ones whose firings
# sit at an IDR's first or last residues, which detect the excision point rather than a motif.

ids = top_features(result, n=TOP_N, drop_boundary=True, feature_dir=bg_fd)
sig_path = write_signature(
    out / "signature.json", {NAME: ids}, case=CASE,
    provenance={"sae": SAE, "positive": POSITIVE, "background": str(bg_path),
                "n_pos": int(n_pos), "n_background": int(n_neg),
                "length_matched": True, "boundary_dropped": True,
                "rank": "log2 odds ratio, descending"})

print(f"wrote {sig_path}")
print(f"{len(ids)} features: {ids[:8]}{' ...' if len(ids) > 8 else ''}")


# What the enriched features detect
# ---------------------------------
# For each feature, take the windows where it fires hardest across the positive set, stack them,
# and render an information-content logo. This is the grammar the signature is made of.

feats, index = sae.encode(POSITIVE, pool="none")
per_seq = per_sequence_activations(feats, index)
print(f"{feats.shape[0]} residues across {len(per_seq)} sequences")

show = ids[:N_FEATURES]
fig, axes = plt.subplots(len(show), 1, figsize=(6, 1.5 * len(show)),
                         constrained_layout=True, squeeze=False)
for ax, fid in zip(axes[:, 0], show):
    windows = top_windows(fid, feats, per_seq, n_windows=N_WINDOWS, half_width=HALF_WIDTH)
    if len(windows) < 2:
        ax.set_axis_off()
        ax.set_title(f"feature {fid}: too few windows", fontsize=8)
        continue
    logomaker.Logo(logomaker.alignment_to_matrix(windows, to_type="information"),
                   ax=ax, color_scheme="chemistry")
    ax.set_title(f"feature {fid}  ({len(windows)} windows)", fontsize=8)
    ax.set_ylabel("bits", fontsize=7)
    ax.set_xticks([])
fig.savefig(out / "logos.png", dpi=200)
print(f"wrote {out / 'logos.png'}")


# Design new sequences carrying this code
# ---------------------------------------
# The signature file is what the sae_only_<name> reward reads. Point the two environment variables
# at it and switch on the RL-SAE term, which ships in configs/grpo.yaml as term 2:
#
#     IDIOM_SAEREWARD_FEATURES=enr/signature.json IDIOM_SAEREWARD_CASE=top30 \
#       idiom_grpo init_from=jxliu2/idiom-300M \
#         reward.terms.2.enabled=true reward.terms.2.reward=sae_only_nucleolus
#
# For a real run use the Slurm template, which sets the same thing:
# sbatch cookbook/slurm/grpo.bash.

print(f"\nnext: IDIOM_SAEREWARD_FEATURES={sig_path} IDIOM_SAEREWARD_CASE={CASE} \\")
print(f"        idiom_grpo init_from=jxliu2/idiom-300M \\")
print(f"          reward.terms.2.enabled=true reward.terms.2.reward=sae_only_{NAME}")
