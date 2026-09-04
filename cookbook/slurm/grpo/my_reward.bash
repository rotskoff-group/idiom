#!/bin/bash
#SBATCH --job-name=idiom-grpo-my-reward
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
# GRPO toward a reward YOU wrote, running in this same interpreter. Use this shape whenever your
# scorer imports in the environment IDiom is installed in -- which is most of the time. A reward
# model needing its own python or torch belongs in one of the grpo/<scorer>.bash scripts instead.
#
# Three ways to name it, all shown below; keep one.
#   1. a file that registers names   -- copy cookbook/rewards/my_rewards.py and edit it
#   2. "module:function"             -- any importable callable, with no decorator and no edit to
#                                       the code defining it. This is the one to use when IDiom was
#                                       pip-installed beside your own package.
#   3. the batched form              -- f(idrs, batch) when scoring a whole step at once is cheaper,
#                                       or when a completion is scored against its GRPO group.
#
# entropy and length are already in the config; reward.add appends on top of them.
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/grpo-my-reward"  # EDIT: keep runs on scratch, not in the repo

# EDIT: pick ONE of these, and comment out the others.
TERM="{reward: fraction_charged, module: $REPO/cookbook/rewards/my_rewards.py, weight: 1.0, shaping: {type: gaussian, target: 0.25, width: 0.5}}"
# TERM="{reward: \"mypackage.scoring:score_idr\", label: mine, weight: 1.0}"
# TERM="{reward: \"mypackage.scoring:score_batch\", label: mine, batched: true, weight: 1.0}"

unset PYTHONPATH PYTHONHOME
# The clone's own venv if `uv sync` made one; otherwise whatever environment idiom is installed in
# (a `pip install git+...` into your own env needs no activation here).
if [[ -f "$REPO/.venv/bin/activate" ]]; then source "$REPO/.venv/bin/activate"; fi
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

idiom_grpo \
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
    reward.add="[$TERM]" \
    trainer.max_steps=3000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    trainer.gradient_clip_val=1.0 \
    trainer.accumulate_grad_batches=2 \
    trainer.log_every_n_steps=1 \
    trainer.checkpoint_every=0 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra"

echo "DONE -> $OUT"
