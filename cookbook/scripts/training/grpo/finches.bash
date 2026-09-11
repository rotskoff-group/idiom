#!/bin/bash

set -euo pipefail

# Design a ProTalpha-like IDR with the native ProTalpha:H1.0 CTD interaction epsilon
# Reference constructs: Ginell et al., Science 2025, Fig. 5 (see example_data/finches)
# FINCHES frontends default to 0.150 M salt, as in the paper's notebook
# Needs: 1 GPU

REPO="/path/to/idiom" # EDIT: repository checkout
OUT="/path/to/output/grpo-finches" # EDIT: run output directory

cd "$REPO"

PARTNER_FASTA=cookbook/example_data/finches/h1_ctd.fasta
PARTNER=$(tail -n +2 "$PARTNER_FASTA" | tr -d '[:space:]') # Skip header and remove whitespace

export PARTNER
export WANDB_MODE=offline

# EDIT: reward terms, scorer commands, shaping, and weights in the matching YAML file
REWARD_FILE="cookbook/scripts/training/grpo/finches.yaml"

# Load the reward YAML with OmegaConf and format its terms for the Hydra override below
REWARD_TERMS=$(python - "$REWARD_FILE" <<'PYTHON'
import json
import sys
from omegaconf import OmegaConf

# Quote string values for Hydra, including commands containing '=' or spaces
def hydra_value(value):
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {hydra_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(hydra_value(item) for item in value) + "]"
    return json.dumps(value)

cfg = OmegaConf.load(sys.argv[1])
terms = OmegaConf.to_container(cfg.reward.terms, resolve=True)
print(hydra_value(terms))
PYTHON
)

# Run training
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
    reward.terms="$REWARD_TERMS" \
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
