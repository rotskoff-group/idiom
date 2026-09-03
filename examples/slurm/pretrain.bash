#!/bin/bash
#SBATCH --job-name=idiom-pretrain
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32          # 4 DataLoader workers per GPU (data.num_workers=4 x 8 GPUs)
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=gpu             # EDIT: your GPU partition
# #SBATCH --account=your_account    # EDIT: uncomment if your site requires an account
# #SBATCH --nodelist=node01         # EDIT: uncomment to pin an 8-GPU node
#SBATCH --output=./slurm_out/slurm-%j.out   # sbatch only; run `mkdir -p slurm_out` first

echo "===== BEGIN SCRIPT: $0 ====="    # save the script verbatim into the slurm log
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SCRIPT: $0 ====="
echo; echo

set -euo pipefail

###
# IDiom 24L "medium" pretrain (~345M params), 1024-token FIM. Global batch = 32/GPU x 8 GPUs x 4 accum
# = 1024. AdamW lr 3e-4, no weight decay, 3k linear warmup -> cosine to 10% over 250k steps; bf16, grad
# clip 1.0, validate every 25k. NO srun: one Slurm task owns the node and Lightning launches one DDP
# process per GPU itself (trainer.devices=8). Lower data.batch_size if you hit OOM.
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/pretrain"     # EDIT: keep runs on scratch, not in the repo
TRAIN_FASTA=/path/to/train.fasta                  # EDIT
VAL_FASTA=/path/to/validation.fasta               # EDIT

unset PYTHONPATH PYTHONHOME                        # use ONLY the uv venv (no module-system leakage)
source "$REPO/.venv/bin/activate"
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# ---- auto-resume from the rolling last.ckpt (restores optimizer/global-step/LR-schedule/RNG) ----
# Fresh run -> no last.ckpt -> empty override -> trains from scratch. On a Slurm timeout, re-submit
# this same script and it picks up where it left off.
CKPT_DIR="$OUT/checkpoints"
RESUME=""
if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  RESUME="resume_from=$CKPT_DIR/last.ckpt"
  echo "RESUMING from $CKPT_DIR/last.ckpt"
fi

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
