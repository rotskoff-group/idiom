#!/bin/bash

set -euo pipefail

###
# GRPO toward a single-chain dimension predicted by sparrow (ALBATROSS); the scorer runs in its own
# uv environment and imports nothing from IDiom. Base generations sit at 26.9 +/- 15.7 A.
# Needs: 1 GPU, ~12 h.
###

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

OUT="${IDIOM_OUT:-$REPO/runs}/grpo-sparrow"

PROPERTY=radius_of_gyration             # EDIT: any ALBATROSS predictor or sequence parameter
TARGET=25                               # EDIT: in the property's units
WIDTH=0.2                               # EDIT: tolerance as a fraction of the target
WEIGHT=0.5                              # EDIT: keep the step-0 contribution near the guardrails
SCORER="uv run --script cookbook/rewards/scorers/sparrow.py --property $PROPERTY"

export WANDB_MODE=offline

# Check the scorer answers before taking the GPU.
python -m idiom.train.grpo.reward.external --cmd "$SCORER"

TERM="{cmd: \"$SCORER\", label: $PROPERTY, weight: $WEIGHT, shaping: {type: quadratic, target: $TARGET, width: $WIDTH}}"

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
