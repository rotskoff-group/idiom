#!/bin/bash
#SBATCH --job-name=idiom-grpo-sae
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

###
# GRPO toward an SAE feature signature (RL-SAE), the paper's headline post-training result.
# sae_only_<name> scores an IDR by the fraction of a target's feature signature firing in it, read
# through a frozen IDiom + SAE. Already a fraction in [0, 1], so no shaping. The reward is library
# code and its signatures ship with it -- no third-party model, no subprocess.
###

REPO=/path/to/idiom                     # EDIT
OUT=/path/to/runs/grpo-sae              # EDIT: keep runs out of the repo

SIGNATURE=nucleolus                     # EDIT: any signature in the targets file
# For your own set, build a signature with cookbook/scripts/feature_enrichment.py and point at it:
# export IDIOM_SAEREWARD_FEATURES=/path/to/signature.json
# export IDIOM_SAEREWARD_CASE=top30

source /path/to/venv/bin/activate       # EDIT
cd "$REPO"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then mkdir -p slurm_out; fi   # #SBATCH --output writes here
export WANDB_MODE=offline               # `wandb login` and set online for live logging

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
