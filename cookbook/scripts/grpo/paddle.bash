#!/bin/bash

set -euo pipefail

# GRPO toward transcriptional activation strength, scored by PADDLE (max Z over 53-residue windows)
# Already a Z-score, so unshaped: base +0.9 (sd 1.7), strong natural ADs 3-8
# Needs: 1 GPU

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-paddle"  # EDIT: run output directory

cd "$REPO"

WEIGHT=0.5                              # EDIT: Z-scale is unbounded above; keep this modest
TIMEOUT=600.0
SCORER="uv run --script cookbook/rewards/scorers/paddle.py"

export WANDB_MODE=offline

python -m idiom.train.grpo.reward.external --cmd "$SCORER"

# Entropy and length terms discourage low-complexity or extreme-length solutions
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: 3.65, width: 0.2}}"

LENGTH="{label: length, \
    weight: 1.0, \
    reward: length, \
    shaping: {name: quadratic, target: 100, width: 1.0}}"

PADDLE="{label: paddle, \
    weight: $WEIGHT, \
    reward: {name: scorer, \
        cmd: \"$SCORER\", \
        timeout: $TIMEOUT, \
        maxlen: 0, \
        cwd: null, \
        env: null, \
        cache_max: 100000, \
        label: null}, \
    shaping: identity}"

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
    reward.terms="[$ENTROPY, $LENGTH, $PADDLE]" \
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
