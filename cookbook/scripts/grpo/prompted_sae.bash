#!/bin/bash

set -euo pipefail

# Redesign one marked IDR using both flanks of a full-length protein as context.
# The bundled P06748_IDR_119-259 record has a 141-residue IDR (1-based, inclusive).
# Only generated IDRs are scored; the native IDR is omitted from the prompt.
# Reward: nucleolus top30 SAE coverage, composition entropy, and native IDR length.
# Needs: 1 GPU

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-prompted-sae"  # EDIT: run output directory

cd "$REPO"

FASTA="$REPO/cookbook/example_data/prompted_grpo/P06748.fasta"  # EDIT: one full protein
TARGET_LENGTH=141                      # EDIT: y - x + 1 for your _IDR_x-y span
SIGNATURE=nucleolus
FEATURES="$REPO/src/idiom/train/grpo/reward/sae_signatures.json"
CASE=top30

# Online logging when authenticated; set WANDB_MODE=offline to save records locally.
export WANDB_MODE="${WANDB_MODE:-online}"

# Entropy and length terms discourage low-complexity or extreme-length solutions
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: 3.65, width: 0.2}}"

LENGTH="{label: length, \
    weight: 1.0, \
    reward: length, \
    shaping: {name: quadratic, target: $TARGET_LENGTH, width: 1.0}}"

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
    run_name=grpo-prompted-nucleolus \
    prompts.mode=prompted \
    prompts.n=1000 \
    prompts.fasta="$FASTA" \
    prompts.n_per=1000 \
    prompts.batch_size=4 \
    grpo.group_size=8 \
    grpo.max_new_tokens=256 \
    grpo.lr=5.0e-6 \
    grpo.beta_kl=0.02 \
    grpo.eps_clip=0.2 \
    grpo.temperature=1.0 \
    grpo.top_k=null \
    grpo.top_p=null \
    grpo.normalize_advantage=true \
    grpo.log_samples_every=5 \
    grpo.n_log_samples=3 \
    grpo.track_disorder=true \
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
    'hydra.sweep.subdir=${hydra.job.num}' \
    "$@"

echo "DONE -> $OUT"
