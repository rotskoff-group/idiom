#!/bin/bash
#SBATCH --job-name=idiom-pretrain
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32          # data.num_workers x trainer.devices
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

###
# Pretrain IDiom 24L (~345M params) on 1024-token FIM. Global batch = 32 x 8 GPUs x 4 accum = 1024.
# No srun: one task owns the node and Lightning launches one process per GPU (trainer.devices=8).
###

REPO=/path/to/idiom                     # EDIT
OUT=/path/to/runs/pretrain              # EDIT: keep runs out of the repo
TRAIN_FASTA=/path/to/train.fasta        # EDIT
VAL_FASTA=/path/to/validation.fasta     # EDIT

source /path/to/venv/bin/activate       # EDIT
cd "$REPO"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then mkdir -p slurm_out; fi   # #SBATCH --output writes here
export WANDB_MODE=offline               # `wandb login` and set online for live logging

# Resume from the rolling checkpoint if one is there; otherwise start fresh.
RESUME=""
if [[ -f "$OUT/checkpoints/last.ckpt" ]]; then RESUME="resume_from=$OUT/checkpoints/last.ckpt"; fi

idiom_train \
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
