#!/bin/bash
#SBATCH --job-name=idiom-grpo
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
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
# GRPO / RL post-training on 1 GPU toward an SAE feature signature (RL-SAE). The reward is a weighted
# sum of terms (entropy + length guardrails + the rl_sae feature-code reward). No third-party reward
# model; runs straight after `uv sync`. init_from takes a HF repo id, a released dir, or a .ckpt.
# For an EXTERNAL reward model (sparrow) instead, set UV_CACHE_DIR and replace the reward.* overrides
# below with a reward.external list (see examples/python/09_sparrow_reward.py and the top-level README).
###

REPO="$SLURM_SUBMIT_DIR"                          # submit from the repo root
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/grpo"         # EDIT: keep runs on scratch, not in the repo
# export UV_CACHE_DIR=/scratch/$USER/uv-cache     # uncomment for an on-demand external reward env

unset PYTHONPATH PYTHONHOME
source "$REPO/.venv/bin/activate"
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---- auto-resume from the rolling last.ckpt (needs trainer.checkpoint_every > 0, set below) ----
CKPT_DIR="$OUT/checkpoints"
RESUME=""
if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  RESUME="resume_from=$CKPT_DIR/last.ckpt"
  echo "RESUMING from $CKPT_DIR/last.ckpt"
fi

idiom_grpo \
    ${RESUME} \
    seed=0 \
    device=auto \
    init_from=jxliu2/idiom-300M \
    prompts.mode=unprompted \
    prompts.n=1000 \
    prompts.fasta=null \
    prompts.n_per=1000 \
    prompts.batch_size=4 \
    grpo.group_size=8 \
    grpo.max_new_tokens=1020 \
    grpo.lr=5.0e-6 \
    grpo.beta_kl=0.02 \
    grpo.eps_clip=0.2 \
    grpo.temperature=1.0 \
    grpo.top_k=null \
    grpo.top_p=null \
    grpo.normalize_advantage=true \
    grpo.log_samples_every=5 \
    grpo.n_log_samples=3 \
    reward.module=null \
    reward.entropy.enabled=true \
    reward.entropy.weight=1.0 \
    reward.entropy.target_entropy=3.65 \
    reward.entropy.width=0.2 \
    reward.length.enabled=true \
    reward.length.weight=1.0 \
    reward.length.target_length=100 \
    reward.length.width=1.0 \
    reward.rl_sae.enabled=true \
    reward.rl_sae.weight=1.0 \
    reward.rl_sae.signature=nucleolus \
    reward.external='[]' \
    trainer.max_steps=3000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    trainer.gradient_clip_val=1.0 \
    trainer.accumulate_grad_batches=2 \
    trainer.log_every_n_steps=1 \
    trainer.checkpoint_every=500 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra"

echo "DONE -> $OUT"
