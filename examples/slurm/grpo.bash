#!/bin/bash
#SBATCH --job-name=idiom-grpo
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu             # EDIT: your GPU partition
# #SBATCH --account=your_account    # EDIT: uncomment if your site requires an account
#SBATCH --output=./slurm_out/slurm-%j.out   # sbatch only; run `mkdir -p slurm_out` first

echo "===== BEGIN SCRIPT: $0 ====="
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SCRIPT: $0 ====="
echo; echo

set -euo pipefail

###
# GRPO / RL post-training on 1 GPU toward an SAE feature signature (RL-SAE). The reward is a weighted
# sum of terms: the entropy and length guardrails plus the SAE feature-code reward. No third-party
# reward model; runs straight after `uv sync`. init_from takes a HF repo id, a released dir, or a
# .ckpt. For a reward model that runs in its OWN environment, use grpo_external.bash instead.
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/grpo"         # EDIT: keep runs on scratch, not in the repo

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
    reward.terms='[{reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}, {reward: length, weight: 1.0, shaping: {type: quadratic, target: 100, width: 1.0}}, {reward: sae_only_nucleolus, module: idiom.train.grpo.reward.rl_sae, weight: 1.0}]' \
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
