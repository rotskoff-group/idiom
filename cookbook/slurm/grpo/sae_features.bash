#!/bin/bash
#SBATCH --job-name=idiom-grpo-sae
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
# GRPO toward an SAE feature signature (RL-SAE) -- the paper's headline post-training result.
#
# sae_only_<name> scores an IDR by the fraction of a target's SAE feature signature firing in it,
# read through a frozen IDiom + SAE lens, so a gain requires encoding the real code rather than
# satisfying a classifier. Already a fraction in [0, 1], so no shaping.
#
# The reward is library code and its signatures ship with it: no third-party model, no subprocess,
# runs straight after install. To design toward your OWN set, build a signature with
# cookbook/scripts/feature_enrichment.py and set IDIOM_SAEREWARD_FEATURES to it.
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/grpo-sae"     # EDIT: keep runs on scratch, not in the repo

SIGNATURE=nucleolus                               # EDIT: any signature in the targets file
# export IDIOM_SAEREWARD_FEATURES=enr/signature.json   # EDIT: uncomment for your own, from feature_enrichment.py
# export IDIOM_SAEREWARD_CASE=top30

unset PYTHONPATH PYTHONHOME
# The clone's own venv if `uv sync` made one; otherwise whatever environment idiom is installed in
# (a `pip install git+...` into your own env needs no activation here).
if [[ -f "$REPO/.venv/bin/activate" ]]; then source "$REPO/.venv/bin/activate"; fi
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# The whole objective, appended to the entropy and length guardrails the config already carries.
TERM="{reward: sae_only_$SIGNATURE, module: idiom.train.grpo.reward.sae_feature, weight: 1.0}"

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
