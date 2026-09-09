#!/bin/bash

set -euo pipefail

# Design a ProTalpha-like IDR with the native ProTalpha:H1.0 CTD interaction epsilon.
# Reference constructs: Ginell et al., Science 2025, Fig. 5 (see example_data/finches).
# FINCHES frontends default to 0.150 M salt, as in the paper's notebook.
# Needs: 1 GPU

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-finches"  # EDIT: run output directory

cd "$REPO"

PARTNER_FASTA=cookbook/example_data/finches/h1_ctd.fasta
PARTNER=$(awk '!/^>/ {gsub(/[[:space:]]/, ""); printf "%s", $0}' "$PARTNER_FASTA")
# Approximate native ProTalpha targets, using Mpipi at 0.150 M salt.
TARGET=-34.8
ENTROPY_TARGET=3.14
LENGTH_TARGET=111
WIDTH=0.2                               # EDIT: tolerance = abs(native epsilon) * WIDTH
WEIGHT=1.0
FORCEFIELD=mpipi                        # Recalculate TARGET if changing force field
SCORER="uv run --script cookbook/rewards/scorers/finches.py --mode heterotypic --partner $PARTNER --forcefield $FORCEFIELD"

export WANDB_MODE=offline

# Match native ProTalpha's entropy and length, allowing its naturally acidic composition.
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: $ENTROPY_TARGET, width: 0.2}}"

LENGTH="{label: length, \
    weight: 1.0, \
    reward: length, \
    shaping: {name: quadratic, target: $LENGTH_TARGET, width: 0.1}}"

EPS="{label: eps, \
    weight: $WEIGHT, \
    reward: {name: scorer, \
        cmd: \"$SCORER\", \
        timeout: 300.0, \
        maxlen: 0, \
        cwd: null, \
        env: null, \
        cache_max: 100000, \
        label: null}, \
    shaping: {name: quadratic, target: $TARGET, width: $WIDTH}}"

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
    grpo.max_new_tokens=256 \
    grpo.lr=5.0e-6 \
    grpo.beta_kl=0.02 \
    grpo.eps_clip=0.2 \
    grpo.temperature=1.0 \
    grpo.top_k=null \
    grpo.top_p=null \
    grpo.normalize_advantage=true \
    grpo.log_samples_every=5 \
    grpo.n_log_samples=3 \
    grpo.track_disorder=true \
    reward.terms="[$ENTROPY, $LENGTH, $EPS]" \
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
