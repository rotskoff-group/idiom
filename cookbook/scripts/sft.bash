#!/bin/bash

set -euo pipefail

###
# Warm-start a released model and specialize it on one curated set (completion-only loss).
# The example set is whole-sequence IDRs with no flanks, so train unprompted (data.prompted_prob=0.0).
# Needs: 1 GPU, ~8 h.
###

# Run in the environment where you pip-installed IDiom; the clone supplies cookbook files.
REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/sft"  # EDIT: run output directory

cd "$REPO"
TRAIN_FASTA=cookbook/example_data/protgps/nucleolus.fasta   # EDIT

export WANDB_MODE=offline

RESUME=""
if [[ -f "$OUT/checkpoints/last.ckpt" ]]; then RESUME="resume_from=$OUT/checkpoints/last.ckpt"; fi

idiom_train_autoreg --config-name sft \
    ${RESUME} \
    seed=0 \
    device=auto \
    init_from=jxliu2/idiom-300M \
    wandb_project=idiom \
    run_name=sft_nucleolus \
    data.train_fasta="$TRAIN_FASTA" \
    data.val_fasta=null \
    data.prompted_prob=0.0 \
    data.completion_only=true \
    data.batch_size=16 \
    data.num_workers=8 \
    optim.lr=1.0e-5 \
    optim.warmup_steps=100 \
    optim.weight_decay=0.1 \
    optim.min_lr_ratio=0.1 \
    trainer.max_steps=1000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=1.0 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra"

echo "DONE -> $OUT/checkpoints/last.ckpt"
