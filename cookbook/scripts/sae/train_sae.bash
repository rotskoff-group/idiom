#!/bin/bash

set -euo pipefail

# Train a top-k SAE, streaming activations from a frozen IDiom (no activation cache on disk)
# layer must be < n_layers; expansion_factor=32 gives 32 x d_model latents
# Needs: 1 GPU

REPO="/path/to/idiom" # EDIT: repository checkout
OUT="/path/to/output/sae" # EDIT: run output directory

cd "$REPO"
FASTA=/path/to/records.fasta # EDIT

export WANDB_MODE=offline

idiom_train_sae \
    seed=0 \
    device=auto \
    model_ckpt=jxliu2/idiom-300M \
    resume_from=null \
    wandb_project=idiom-sae \
    run_name=null \
    layer=18 \
    region=all \
    data.fasta="$FASTA" \
    data.prompted_prob=0.5 \
    data.shuffle=true \
    data.record_batch_size=16 \
    sae_batch_size=4096 \
    buffer_size=262144 \
    init_b_dec_from_mean=true \
    sae.k=32 \
    sae.expansion_factor=32 \
    sae.activation=topk \
    sae.multi_topk=false \
    sae.auxk_alpha=0.03125 \
    sae.dead_feature_tokens=10000000 \
    sae.warmup_steps=1000 \
    trainer.max_steps=100000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra" \
    hydra.sweep.dir="$OUT/hydra/multirun" \
    'hydra.sweep.subdir=${hydra.job.num}'

echo "DONE -> $OUT"
