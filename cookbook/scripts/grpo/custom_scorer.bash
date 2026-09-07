#!/bin/bash

set -euo pipefail

###
# GRPO toward a reward model you wrote, running in its own environment. Copy the template at
# cookbook/rewards/scorers/custom_scorer.py. Use this only when its deps cannot coexist with IDiom's.
# Needs: 1 GPU, ~8 h.
###

# Run in the environment where you pip-installed IDiom; the clone supplies cookbook files.
REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-my-scorer"  # EDIT: run output directory

cd "$REPO"

PROPERTY=isoelectric_point              # EDIT: isoelectric_point | molecular_weight
TARGET=4.5                              # EDIT: acidic
WIDTH=0.3                               # EDIT: fraction of the target
WEIGHT=1.0
SCORER="uv run --script cookbook/rewards/scorers/custom_scorer.py --property $PROPERTY"

export WANDB_MODE=offline

# Check the scorer answers before taking the GPU.
python -m idiom.train.grpo.reward.external --cmd "$SCORER"

# The whole objective, written out: nothing is added for you and reward.terms is empty by
# default. entropy and length keep the target from being met by a low-complexity tract or a
# degenerate length; drop either line and it is gone.
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: 3.65, width: 0.2}}"

LENGTH="{label: length, \
    weight: 1.0, \
    reward: length, \
    shaping: {name: quadratic, target: 100, width: 1.0}}"

MINE="{label: $PROPERTY, \
    weight: $WEIGHT, \
    reward: {name: scorer, \
        cmd: \"$SCORER\", \
        timeout: 300.0, \
        maxlen: 0, \
        cwd: null, \
        env: null, \
        cache_max: 100000, \
        label: null}, \
    shaping: {name: gaussian, target: $TARGET, width: $WIDTH}}"

idiom_train_grpo \
    seed=0 \
    device=auto \
    init_from=jxliu2/idiom-300M \
    resume_from=null \
    wandb_project=idiom-grpo \
    run_name=null \
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
    reward.terms="[$ENTROPY, $LENGTH, $MINE]" \
    trainer.max_steps=3000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    trainer.gradient_clip_val=1.0 \
    trainer.accumulate_grad_batches=2 \
    trainer.log_every_n_steps=1 \
    trainer.checkpoint_every=0 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra" \
    hydra.sweep.dir="$OUT/hydra/multirun" \
    'hydra.sweep.subdir=${hydra.job.num}'

echo "DONE -> $OUT"
