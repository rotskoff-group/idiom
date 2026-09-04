#!/bin/bash
#SBATCH --job-name=idiom-grpo-paddle
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

###
# GRPO toward transcriptional activation strength, scored by PADDLE (max Z over 53-residue windows).
# PADDLE-noSS runs from sequence alone; uv builds its TensorFlow environment from the script header.
#
# The raw reward is a Z-score against PADDLE's background, so it is already on a usable scale: base
# generations sit at +0.9 (sd 1.7), strong natural activation domains at 3-8. Left unshaped to
# maximize; the weight is what keeps it from dwarfing the guardrails. ~3 s per step.
###

REPO=/path/to/idiom                     # EDIT
OUT=/path/to/runs/grpo-paddle           # EDIT: keep runs out of the repo

WEIGHT=0.5                              # EDIT: Z-scale is unbounded above; keep this modest
TIMEOUT=600.0
SCORER="uv run --script cookbook/rewards/scorers/paddle.py"

source /path/to/venv/bin/activate       # EDIT
cd "$REPO"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then mkdir -p slurm_out; fi   # #SBATCH --output writes here
export WANDB_MODE=offline               # `wandb login` and set online for live logging
export UV_CACHE_DIR=/path/to/uv-cache   # EDIT: an on-demand env is several GB

# Build the scorer's environment and check it answers, before taking the GPU. The first build
# downloads and compiles; every run after is a uv cache hit.
python -m idiom.train.grpo.reward.external --cmd "$SCORER"

# The whole objective, appended to the entropy and length guardrails the config already carries.
TERM="{cmd: \"$SCORER\", label: paddle, weight: $WEIGHT, timeout: $TIMEOUT}"

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
