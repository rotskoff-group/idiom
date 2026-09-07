#!/bin/bash

set -euo pipefail

# Pretrain IDiom 24L (~302M params) on 1024-token FIM. Global batch = 32 x 8 GPUs x 4 accum = 1024.
# Training settings match src/idiom/configs/pretrain.yaml
# Single node: launch this script once; Lightning starts one process per GPU
# For multi-node pretraining, use the srun example in cookbook/scripts/README.md
# Needs: 8 GPUs

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/pretrain"  # EDIT: run output directory

cd "$REPO"
TRAIN_FASTA=/path/to/train.fasta        # EDIT
VAL_FASTA=/path/to/validation.fasta     # EDIT

export WANDB_MODE=offline

RESUME=null
if [[ -f "$OUT/checkpoints/last.ckpt" ]]; then RESUME="$OUT/checkpoints/last.ckpt"; fi

idiom_train_autoreg \
    seed=0 \
    device=auto \
    init_from=null \
    resume_from="$RESUME" \
    ckpt_every_n_steps=2000 \
    wandb_project=idiom \
    run_name=pretrain_24L \
    data.train_fasta="$TRAIN_FASTA" \
    data.val_fasta="$VAL_FASTA" \
    data.prompted_prob=0.5 \
    data.completion_only=false \
    data.batch_size=32 \
    data.num_workers=8 \
    model.vocab_size=27 \
    model.n_layers=24 \
    model.d_model=1024 \
    model.n_heads=16 \
    model.max_seq_len=1024 \
    model.rope_base=10000.0 \
    model.expansion_ratio=2.6666666666666665 \
    model.norm_eps=1.0e-5 \
    model.qk_norm=true \
    model.tie_embeddings=true \
    optim.lr=4.0e-4 \
    optim.warmup_steps=3000 \
    optim.weight_decay=0.0 \
    'optim.betas=[0.9,0.95]' \
    optim.min_lr_ratio=0.1 \
    trainer.max_steps=250000 \
    trainer.accelerator=auto \
    trainer.devices=8 \
    trainer.precision=bf16-mixed \
    trainer.val_check_interval=25000 \
    trainer.gradient_clip_val=0.0 \
    trainer.accumulate_grad_batches=4 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra" \
    hydra.sweep.dir="$OUT/hydra/multirun" \
    'hydra.sweep.subdir=${hydra.job.num}'

echo "DONE -> $OUT/checkpoints/last.ckpt"
