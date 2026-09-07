#!/bin/bash

set -euo pipefail

# Optimize ProtGPS compartment probabilities without shaping.
# Needs: 1 GPU, ~12 h.

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-protgps"  # EDIT: run output directory

cd "$REPO"

COMPARTMENT=nucleolus                   # EDIT: any of the 12, or max / mean
WEIGHT=1.0
SCORER="uv run --script cookbook/rewards/scorers/protgps.py --compartment $COMPARTMENT"

export WANDB_MODE=offline
# The pinned CUDA build fails on H100; batching can change a sequence's probability.
export IDIOM_PROTGPS_DEVICE=cpu
export PROTGPS_BATCH=1

python -m idiom.train.grpo.reward.external --cmd "$SCORER"

# ProtGPS is length-sensitive; this objective uses entropy regularization without a length term.
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: 3.65, width: 0.2}}"

PROTGPS="{label: protgps, \
    weight: $WEIGHT, \
    reward: {name: scorer, \
        cmd: \"$SCORER\", \
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
    reward.terms="[$ENTROPY, $PROTGPS]" \
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
