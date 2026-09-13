#!/bin/bash

set -euo pipefail

# Train the paper's layer-18 SAE on unprompted IDR activations from frozen IDiom-300M
# Zero-based layer 18; expansion_factor=16 gives 16,384 latents and an auto-scaled LR of 2e-4
# Needs: 1 GPU

REPO="/path/to/idiom" # EDIT: repository checkout
OUT="/path/to/output/sae" # EDIT: run output directory

cd "$REPO"
FASTA=/path/to/train.fasta # EDIT: pretraining train-split record FASTA

export WANDB_MODE=offline

idiom_train_sae \
    seed=0 \
    device=auto \
    model_ckpt=jxliu2/idiom-300M \
    resume_from=null \
    wandb_project=idiom-sae \
    run_name=null \
    layer=18 \
    region=idr \
    data.fasta="$FASTA" \
    data.prompted_prob=0.0 \
    data.shuffle=true \
    data.record_batch_size=16 \
    sae_batch_size=4096 \
    buffer_size=131072 \
    init_b_dec_from_mean=true \
    sae.k=32 \
    sae.expansion_factor=16 \
    sae.activation=topk \
    sae.multi_topk=false \
    sae.auxk_alpha=0.03125 \
    sae.dead_feature_tokens=10000000 \
    sae.warmup_steps=1000 \
    trainer.max_steps=50000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra" \
    hydra.sweep.dir="$OUT/hydra/multirun" \
    'hydra.sweep.subdir=${hydra.job.num}'

echo "DONE -> $OUT"
