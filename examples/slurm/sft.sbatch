#!/bin/bash
#SBATCH --job-name=idiom-sft
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8           # DataLoader workers (data.num_workers=8)
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu             # EDIT: your GPU partition
# #SBATCH --account=your_account    # EDIT: uncomment if your site requires an account
#SBATCH --output=./slurm_out/slurm-%j.out   # run `mkdir -p slurm_out` once before submitting

echo "===== BEGIN SLURM SCRIPT: $0 ====="
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SLURM SCRIPT: $0 ====="
echo; echo

set -euo pipefail

###
# SFT on 1 GPU: warm-start a released model and specialize it on one curated set (completion-only
# loss). init_from takes a HF repo id, a released dir, or a .ckpt. The ProtGPS/effector sets are
# whole-sequence IDRs (no flanks), so train the unprompted form (data.prompted_prob=0.0).
###

REPO="$SLURM_SUBMIT_DIR"                          # submit from the repo root
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/sft"          # EDIT: keep runs on scratch, not in the repo

unset PYTHONPATH PYTHONHOME
source "$REPO/.venv/bin/activate"
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---- auto-resume from the rolling last.ckpt ----
CKPT_DIR="$OUT/checkpoints"
RESUME=""
if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  RESUME="resume_from=$CKPT_DIR/last.ckpt"
  echo "RESUMING from $CKPT_DIR/last.ckpt"
fi

idiom_train --config-name sft \
    ${RESUME} \
    seed=0 \
    device=auto \
    init_from=jxliu2/idiom-300M \
    wandb_project=idiom \
    run_name=sft_nucleolus \
    data.train_fasta=examples/example_data/protgps/nucleolus.fasta \
    data.val_fasta=null \
    data.prompted_prob=0.0 \
    data.completion_only=true \
    data.batch_size=32 \
    data.num_workers=8 \
    optim.lr=1.0e-5 \
    optim.warmup_steps=100 \
    optim.weight_decay=0.1 \
    optim.min_lr_ratio=0.1 \
    trainer.max_steps=2000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=1.0 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra"

echo "DONE -> $OUT/checkpoints/last.ckpt"
