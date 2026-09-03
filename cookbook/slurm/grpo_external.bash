#!/bin/bash
#SBATCH --job-name=idiom-grpo-ext
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
# GRPO / RL post-training on 1 GPU toward an EXTERNAL reward model, run in its own environment.
# The scorer here is sparrow (biophysics: radius of gyration, asphericity, kappa, ...); it imports
# nothing from IDiom and declares its own dependencies in its script header, so uv builds and
# caches that environment on demand. Swap the cmd for any program speaking the scorer protocol --
# a pre-built venv, a conda env, a container. For the built-in SAE reward instead, use grpo.bash.
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/grpo-external"  # EDIT: keep runs on scratch, not in the repo
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/uv-cache}"  # EDIT: an on-demand env is several GB

PROPERTY=radius_of_gyration                       # EDIT: any sparrow property
TARGET=25                                         # EDIT: in the property's units
WIDTH=0.2                                         # EDIT: tolerance as a fraction of the target
SCORER="uv run --script $(python -c 'from idiom.rewards import rewards_path; print(rewards_path("external_rewards/sparrow.py"))') --property $PROPERTY"

unset PYTHONPATH PYTHONHOME
source "$REPO/.venv/bin/activate"
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---- pre-flight: build the scorer environment and check it answers, before taking the GPU ----
# The first build is slow (sparrow needs a C compiler); every run after is a uv cache hit. A broken
# command fails here in seconds instead of after the policy has warm-started.
python -m idiom.train.grpo.reward.external \
    --cmd "$SCORER" --shaping quadratic --target "$TARGET" --width "$WIDTH"

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
    reward.terms="[{reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}, {reward: length, weight: 1.0, shaping: {type: quadratic, target: 100, width: 1.0}}, {label: $PROPERTY, weight: 1.0, cmd: '$SCORER', shaping: {type: quadratic, target: $TARGET, width: $WIDTH}}]" \
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
