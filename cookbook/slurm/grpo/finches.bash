#!/bin/bash
#SBATCH --job-name=idiom-grpo-finches
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
# GRPO toward attractive interaction chemistry, scored by FINCHES (epsilon, from a coarse-grained
# force field). Negative epsilon is attractive, so a negative target designs self-attractive,
# condensate-prone IDRs.
#
# Base generations average +3.6 (sd 7.0); an FUS-LC-like aromatic tract is about -8.5. The width
# below keeps a starting sequence about one unit under, rather than swamping the guardrails: aimed
# at -6.0 with width 0.3 the term starts at -31.5 and drowns everything, which is exactly the
# failure this file is calibrated to avoid.
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/grpo-finches"   # EDIT: keep runs on scratch, not in the repo

MODE=homotypic                                    # EDIT: homotypic, or heterotypic with --partner
TARGET=-6.0                                       # EDIT: epsilon; negative is attractive
WIDTH=1.0                                         # EDIT: absolute here, since epsilon crosses zero
WEIGHT=1.0
SCORER="uv run --script $REPO/cookbook/rewards/scorers/finches.py --mode $MODE"

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
TERM="{cmd: \"$SCORER\", label: eps, weight: $WEIGHT, shaping: {type: quadratic, target: $TARGET, width: $WIDTH}}"

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
