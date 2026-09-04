#!/bin/bash
#SBATCH --job-name=idiom-grpo-my-reward
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

###
# GRPO toward a reward YOU wrote, running in this same interpreter -- the right shape whenever your
# scorer imports in the environment IDiom is installed in. A reward model needing its own python or
# torch belongs in one of the other scripts here instead.
#
# Three ways to name it, all below; keep one. entropy and length are already in the config;
# reward.add appends on top of them.
###

REPO=/path/to/idiom                     # EDIT
OUT=/path/to/runs/grpo-my-reward        # EDIT: keep runs out of the repo

# EDIT: pick ONE.
TERM="{reward: fraction_charged, module: cookbook/rewards/my_rewards.py, weight: 1.0, shaping: {type: gaussian, target: 0.25, width: 0.5}}"
# TERM="{reward: \"mypackage.scoring:score_idr\", label: mine, weight: 1.0}"                    # any importable callable
# TERM="{reward: \"mypackage.scoring:score_batch\", label: mine, batched: true, weight: 1.0}"   # f(idrs, batch), scored a step at a time

source /path/to/venv/bin/activate       # EDIT
cd "$REPO"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then mkdir -p slurm_out; fi   # #SBATCH --output writes here
export WANDB_MODE=offline               # `wandb login` and set online for live logging

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
