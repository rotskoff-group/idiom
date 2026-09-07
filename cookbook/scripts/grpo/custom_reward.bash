#!/bin/bash

set -euo pipefail

###
# GRPO toward a reward you wrote, running in this interpreter. Four ways to name it below; keep one.
# The last shapes it with a rule of your own; both live in custom_rewards.py and are named by path,
# so nothing needs registering. If it needs its own python or torch, use custom_scorer.bash instead.
# Needs: 1 GPU, ~12 h.
###

# Run in the environment where you pip-installed IDiom; the clone supplies cookbook files.
REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-my-reward"  # EDIT: run output directory

cd "$REPO"

# The whole objective, written out: nothing is added for you and reward.terms is empty by default.
# This one is deliberately bare -- a single term, no entropy and no length -- to show that a run
# gets exactly the objective it names. Uncomment the two below for a real run: without them the
# target is satisfiable by a low-complexity tract or a degenerate length.
# ENTROPY='{label: entropy, weight: 1.0, reward: entropy, shaping: {name: quadratic, target: 3.65, width: 0.2}}'
# LENGTH='{label: length,  weight: 1.0, reward: length,  shaping: {name: quadratic, target: 100,  width: 1.0}}'

# EDIT: pick ONE.
MINE="{label: fcr, \
    weight: 1.0, \
    reward: \"cookbook/rewards/custom_rewards.py:fraction_charged\", \
    shaping: {name: gaussian, target: 0.25, width: 0.5}}"
# MINE="{label: mine, weight: 1.0, reward: \"mypackage.scoring:score_idr\", shaping: identity}"                # any importable factory
# MINE="{label: mine, weight: 1.0, reward: {name: \"mypackage.scoring:make_scorer\", cutoff: 0.3}, shaping: identity}"   # a factory with arguments
# MINE="{label: fcr, weight: 1.0, reward: \"cookbook/rewards/custom_rewards.py:fraction_charged\", shaping: {name: \"cookbook/rewards/custom_rewards.py:one_sided\", target: 0.30, width: 0.5}}"   # shaping of your own

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
    reward.terms="[$MINE]" \
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
