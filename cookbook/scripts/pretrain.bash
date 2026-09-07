#!/bin/bash

set -euo pipefail

###
# Pretrain IDiom 24L (~345M params) on 1024-token FIM. Global batch = 32 x 8 GPUs x 4 accum = 1024.
# Lightning launches one process per GPU itself; do not wrap this in a launcher.
# Needs: 8 GPUs, ~7 days.
###

# Run in the environment where you pip-installed IDiom; the clone supplies cookbook files.
REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/pretrain"  # EDIT: run output directory

cd "$REPO"
TRAIN_FASTA=/path/to/train.fasta        # EDIT
VAL_FASTA=/path/to/validation.fasta     # EDIT

export WANDB_MODE=offline

RESUME=""
if [[ -f "$OUT/checkpoints/last.ckpt" ]]; then RESUME="resume_from=$OUT/checkpoints/last.ckpt"; fi

idiom_train_autoreg \
    ${RESUME} \
    seed=0 \
    device=auto \
    init_from=null \
    ckpt_every_n_steps=1000 \
    wandb_project=idiom \
    run_name=pretrain_24L \
    data.train_fasta="$TRAIN_FASTA" \
    data.val_fasta="$VAL_FASTA" \
    data.prompted_prob=0.5 \
    data.completion_only=false \
    data.batch_size=32 \
    data.num_workers=4 \
    model.n_layers=24 \
    model.d_model=1024 \
    model.n_heads=16 \
    model.max_seq_len=1024 \
    optim.lr=3.0e-4 \
    optim.warmup_steps=3000 \
    optim.weight_decay=0.0 \
    optim.min_lr_ratio=0.1 \
    trainer.max_steps=250000 \
    trainer.accelerator=auto \
    trainer.devices=8 \
    trainer.precision=bf16-mixed \
    trainer.val_check_interval=25000 \
    trainer.gradient_clip_val=1.0 \
    trainer.accumulate_grad_batches=4 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra"

echo "DONE -> $OUT/checkpoints/last.ckpt"
