#!/bin/bash

set -euo pipefail

###
# GRPO toward a condensate compartment, scored by ProtGPS. Its environment pins python 3.8 and
# torch 2.0, which is why it runs out of process. Already a probability, so no shaping.
# Needs: 1 GPU, ~12 h.
###

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

OUT="${IDIOM_OUT:-$REPO/runs}/grpo-protgps"

COMPARTMENT=nucleolus                   # EDIT: any of the 12, or max / mean
WEIGHT=1.0
SCORER="uv run --script cookbook/rewards/scorers/protgps.py --compartment $COMPARTMENT"

export WANDB_MODE=offline

# Check the scorer answers before taking the GPU.
python -m idiom.train.grpo.reward.external --cmd "$SCORER"

# The whole objective, written out: nothing is added for you and reward.terms is empty by
# default. entropy and length keep the target from being met by a low-complexity tract or a
# degenerate length; drop either line and it is gone.
# A probability in [0, 1], and compartment prediction is length-sensitive on its own, so this
# objective carries entropy and no length term.
ENTROPY='{reward: entropy, weight: 1.0, shaping: {type: quadratic, target: 3.65, width: 0.2}}'
PROTGPS="{cmd: \"$SCORER\", label: protgps, weight: $WEIGHT}"

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
    reward.terms="[$ENTROPY, $PROTGPS]" \
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
