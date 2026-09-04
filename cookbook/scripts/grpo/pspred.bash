#!/bin/bash

set -euo pipefail

###
# GRPO toward phase-separation thermodynamics, scored by PSpred. dG is transfer free energy in kT.
# Base generations average -0.1 (sd 0.8); LAF1, a 170-residue LLPS driver, reaches -6.1.
# Needs: 1 GPU, ~12 h.
###

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

OUT="${IDIOM_OUT:-$REPO/runs}/grpo-pspred"

TARGET_KIND=dG                          # EDIT: dG | logcdil_mgml | cdil_mgml
TARGET=-3.0                             # EDIT: in kT for dG
WIDTH=1.0                               # EDIT: absolute, since dG crosses zero
WEIGHT=1.0
SCORER="uv run --script cookbook/rewards/scorers/pspred.py --target $TARGET_KIND"

export WANDB_MODE=offline

# Check the scorer answers before taking the GPU.
python -m idiom.train.grpo.reward.external --cmd "$SCORER"

TERM="{cmd: \"$SCORER\", label: $TARGET_KIND, weight: $WEIGHT, shaping: {type: quadratic, target: $TARGET, width: $WIDTH}}"

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
