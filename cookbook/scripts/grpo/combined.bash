#!/bin/bash

set -euo pipefail

###
# GRPO toward TWO reward models at once: attractive interaction chemistry (FINCHES) and a condensate
# compartment (ProtGPS). Terms compose in the order named, and each keeps its own scorer process.
# The objective here carries entropy and no length term, to show that both are choices a run makes.
# Needs: 1 GPU, ~16 h.
###

# Run in the environment where you pip-installed IDiom; the clone supplies cookbook files.
REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-combined"  # EDIT: run output directory

cd "$REPO"

EPS_TARGET=-6.0                         # EDIT: epsilon; negative is attractive
EPS_WIDTH=1.0                           # EDIT: tolerance = abs(EPS_TARGET) * EPS_WIDTH (6.0 here)
EPS_WEIGHT=0.5                          # EDIT: two objectives share one budget; see rewards/README.md
COMPARTMENT=nucleolus                   # EDIT: any of the 12, or max / mean
PROTGPS_WEIGHT=1.0

FINCHES="uv run --script cookbook/rewards/scorers/finches.py --mode homotypic --forcefield mpipi"
PROTGPS="uv run --script cookbook/rewards/scorers/protgps.py --compartment $COMPARTMENT"

export WANDB_MODE=offline

# Check both scorers answer before taking the GPU.
python -m idiom.train.grpo.reward.external --cmd "$FINCHES"
python -m idiom.train.grpo.reward.external --cmd "$PROTGPS"

# The whole objective, written out: nothing is added for you and reward.terms is empty by default.
# ProtGPS returns a probability and is length-sensitive on its own, so this run keeps entropy and
# leaves length out; add a length term back if the generations drift long.
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: 3.65, width: 0.2}}"

EPS="{label: eps, \
    weight: $EPS_WEIGHT, \
    reward: {name: scorer, \
        cmd: \"$FINCHES\", \
        timeout: 300.0, \
        maxlen: 0, \
        cwd: null, \
        env: null, \
        cache_max: 100000, \
        label: null}, \
    shaping: {name: quadratic, target: $EPS_TARGET, width: $EPS_WIDTH}}"

COMPARTMENT_TERM="{label: protgps, \
    weight: $PROTGPS_WEIGHT, \
    reward: {name: scorer, \
        cmd: \"$PROTGPS\", \
        timeout: 300.0, \
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
    reward.terms="[$ENTROPY, $EPS, $COMPARTMENT_TERM]" \
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
