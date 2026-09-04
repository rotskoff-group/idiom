#!/bin/bash
#SBATCH --job-name=idiom-grpo-protgps
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

###
# GRPO toward a subcellular condensate compartment, scored by ProtGPS (an ESM-2 classifier). This is
# the scorer that shows why external rewards exist: its environment pins python 3.8, torch 2.0 and
# pytorch-lightning 1.6.4, none of which can coexist with IDiom.
#
# The raw reward is already a probability in [0, 1], so no shaping. ProtGPS rewards compositional
# extremity and high scores are reachable with low-complexity tracts, so keep the guardrails on.
###

REPO=/path/to/idiom                     # EDIT
OUT=/path/to/runs/grpo-protgps          # EDIT: keep runs out of the repo

COMPARTMENT=nucleolus                   # EDIT: any of the 12, or max / mean
WEIGHT=1.0
SCORER="uv run --script cookbook/rewards/scorers/protgps.py --compartment $COMPARTMENT"

source /path/to/venv/bin/activate       # EDIT
cd "$REPO"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then mkdir -p slurm_out; fi   # #SBATCH --output writes here
export WANDB_MODE=offline               # `wandb login` and set online for live logging
export UV_CACHE_DIR=/path/to/uv-cache   # EDIT: an on-demand env is several GB

# Build the scorer's environment and check it answers, before taking the GPU. The first build
# downloads and compiles; every run after is a uv cache hit.
python -m idiom.train.grpo.reward.external --cmd "$SCORER"

# The whole objective, appended to the entropy and length guardrails the config already carries.
TERM="{cmd: \"$SCORER\", label: protgps, weight: $WEIGHT}"

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
