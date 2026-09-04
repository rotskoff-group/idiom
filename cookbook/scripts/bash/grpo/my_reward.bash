#!/bin/bash

set -euo pipefail

###
# GRPO toward a reward you wrote, running in this interpreter. Three ways to name it below; keep one.
# If it needs its own python or torch, use my_scorer.bash instead.
# Needs: 1 GPU, ~12 h.
###

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

OUT="${IDIOM_OUT:-$REPO/runs}/grpo-my-reward"

# EDIT: pick ONE.
TERM="{reward: fraction_charged, module: cookbook/rewards/my_rewards.py, weight: 1.0, shaping: {type: gaussian, target: 0.25, width: 0.5}}"
# TERM="{reward: \"mypackage.scoring:score_idr\", label: mine, weight: 1.0}"                    # any importable callable
# TERM="{reward: \"mypackage.scoring:score_batch\", label: mine, batched: true, weight: 1.0}"   # f(idrs, batch), scored a step at a time

export WANDB_MODE=offline

idiom_train_grpo \
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
