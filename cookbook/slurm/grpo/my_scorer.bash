#!/bin/bash
#SBATCH --job-name=idiom-grpo-my-scorer
#SBATCH --time=08:00:00
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
# GRPO toward a reward model YOU wrote, running in its own environment.
#
# The scorer here is cookbook/rewards/scorers/my_scorer.py, the template: about 40 lines that speak
# the JSON protocol and compute isoelectric point from Biopython. Copy it, change what it computes,
# and point SCORER at your copy.
#
# Use this shape only when your scorer's dependencies cannot coexist with IDiom's. If it imports in
# IDiom's environment, grpo/my_reward.bash is simpler and much faster -- a subprocess to call a
# function you could have imported is pure overhead.
#
# pI is a real handle on IDR behaviour: most nuclear IDRs are acidic, and pI tracks the charge
# balance that drives complex coacervation.
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/grpo-my_scorer"   # EDIT: keep runs on scratch, not in the repo

PROPERTY=isoelectric_point                        # EDIT: isoelectric_point | molecular_weight
TARGET=4.5                                        # EDIT: acidic
WIDTH=0.3                                         # EDIT: fraction of the target
WEIGHT=1.0
SCORER="uv run --script $REPO/cookbook/rewards/scorers/my_scorer.py --property $PROPERTY"

unset PYTHONPATH PYTHONHOME
# The clone's own venv if `uv sync` made one; otherwise whatever environment idiom is installed in
# (a `pip install git+...` into your own env needs no activation here).
if [[ -f "$REPO/.venv/bin/activate" ]]; then source "$REPO/.venv/bin/activate"; fi
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/uv-cache}"   # EDIT: an on-demand env is several GB
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---- pre-flight: build the scorer's environment and check it answers, before taking the GPU ----
# The first build downloads and compiles; every run after is a uv cache hit. A broken command fails
# here in seconds instead of after the policy has warm-started.
python -m idiom.train.grpo.reward.external --cmd "$SCORER"

# The whole objective, appended to the entropy and length guardrails the config already carries.
TERM="{cmd: \"$SCORER\", label: $PROPERTY, weight: $WEIGHT, shaping: {type: gaussian, target: $TARGET, width: $WIDTH}}"

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
