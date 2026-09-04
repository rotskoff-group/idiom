#!/bin/bash
#SBATCH --job-name=idiom-sft
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

###
# Warm-start a released model and specialize it on one curated set (completion-only loss).
# init_from takes a HF repo id, a released dir, or a .ckpt. The example set is whole-sequence
# IDRs with no flanks, so train the unprompted form (data.prompted_prob=0.0).
###

REPO=/path/to/idiom                     # EDIT
OUT=/path/to/runs/sft                   # EDIT: keep runs out of the repo
TRAIN_FASTA=cookbook/example_data/protgps/nucleolus.fasta   # EDIT

source /path/to/venv/bin/activate       # EDIT
cd "$REPO"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then mkdir -p slurm_out; fi   # #SBATCH --output writes here
export WANDB_MODE=offline               # `wandb login` and set online for live logging

RESUME=""
if [[ -f "$OUT/checkpoints/last.ckpt" ]]; then RESUME="resume_from=$OUT/checkpoints/last.ckpt"; fi

idiom_train --config-name sft \
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
