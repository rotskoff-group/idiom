#!/bin/bash

set -euo pipefail

# GRPO toward an SAE feature signature (RL-SAE). Scores the fraction of a signature firing in an
# IDR; already in [0, 1], so no shaping. Run the feature_enrichment notebook first to write the signature.
# Needs: 1 GPU

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-sae"  # EDIT: run output directory

cd "$REPO"

SIGNATURE=nucleolus                     # EDIT: the NAME the feature_enrichment notebook wrote

# The signature JSON the feature_enrichment notebook writes
FEATURES="/path/to/signature.json"       # EDIT: notebook output
CASE=top30                              # EDIT: the CASE the feature_enrichment notebook wrote

export WANDB_MODE=offline

if [[ ! -f "$FEATURES" ]]; then
    echo "no signature at $FEATURES -- run the cookbook/notebooks/feature_enrichment.ipynb" >&2
    echo "notebook first, then set FEATURES to the saved signature.json path." >&2
    exit 1
fi

# Entropy and length terms discourage low-complexity or extreme-length solutions
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: 3.65, width: 0.2}}"

LENGTH="{label: length, \
    weight: 1.0, \
    reward: length, \
    shaping: {name: quadratic, target: 100, width: 1.0}}"

SAE="{label: sae, \
    weight: 1.0, \
    reward: {name: sae_signature, \
        signature: $SIGNATURE, \
        features: \"$FEATURES\", \
        case: $CASE, \
        sae: jxliu2/idiomsae-300M-L18-k32, \
        device: null}, \
    shaping: identity}"

idiom_train_grpo \
    seed=0 \
    device=auto \
    init_from=jxliu2/idiom-300M \
    resume_from=null \
    wandb_project=idiom-grpo \
    run_name=null \
    prompts.mode=unprompted \
    prompts.n=1000 \
    prompts.fasta=null \
    prompts.n_per=1000 \
    prompts.batch_size=4 \
    grpo.group_size=8 \
    grpo.max_new_tokens=1020 \
    grpo.lr=5.0e-6 \
    grpo.beta_kl=0.02 \
    grpo.eps_clip=0.2 \
    grpo.temperature=1.0 \
    grpo.top_k=null \
    grpo.top_p=null \
    grpo.normalize_advantage=true \
    grpo.log_samples_every=5 \
    grpo.n_log_samples=3 \
    reward.terms="[$ENTROPY, $LENGTH, $SAE]" \
    trainer.max_steps=3000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    trainer.gradient_clip_val=1.0 \
    trainer.accumulate_grad_batches=2 \
    trainer.log_every_n_steps=1 \
    trainer.checkpoint_every=0 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra" \
    hydra.sweep.dir="$OUT/hydra/multirun" \
    'hydra.sweep.subdir=${hydra.job.num}'

echo "DONE -> $OUT"
