#!/bin/bash

set -euo pipefail

# Redesign one marked IDR using both flanks of a full-length protein as context.
# The bundled P45973_IDR_79-123 record has a 45-residue hinge/linker (1-based, inclusive).
# Only generated IDRs are scored; the native IDR is omitted from the prompt.
# Reward: linker-only ALBATROSS Rg, composition entropy, and native linker length.
# Rg is predicted for the isolated linker, not for the full protein or domain separation.
# Needs: 1 GPU

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/grpo-prompted-linker-rg"  # EDIT: run output directory

cd "$REPO"

FASTA="$REPO/cookbook/example_data/prompted_grpo/P45973.fasta"  # EDIT: one full protein
TARGET_LENGTH=45                      # EDIT: y - x + 1 for your _IDR_x-y span
TARGET_RG=25                           # EDIT: illustrative target in angstroms
RG_WIDTH=0.2                           # quadratic reward scale = 20% of target Rg
LENGTH_WIDTH=0.1                       # discourage reaching Rg by changing linker length
SCORER="uv run --script cookbook/rewards/scorers/sparrow.py --property radius_of_gyration"

export WANDB_MODE=offline

# Entropy and length terms discourage low-complexity or extreme-length solutions
ENTROPY="{label: entropy, \
    weight: 1.0, \
    reward: entropy, \
    shaping: {name: quadratic, target: 3.65, width: 0.2}}"

LENGTH="{label: length, \
    weight: 1.0, \
    reward: length, \
    shaping: {name: quadratic, target: $TARGET_LENGTH, width: $LENGTH_WIDTH}}"

RG="{label: rg, \
    weight: 0.5, \
    reward: {name: scorer, \
        cmd: \"$SCORER\", \
        timeout: 300.0, \
        maxlen: 0, \
        cwd: null, \
        env: null, \
        cache_max: 100000, \
        label: null}, \
    shaping: {name: quadratic, target: $TARGET_RG, width: $RG_WIDTH}}"

idiom_train_grpo \
    seed=0 \
    device=auto \
    init_from=jxliu2/idiom-300M \
    resume_from=null \
    wandb_project=idiom-grpo \
    run_name=grpo-prompted-linker-rg \
    prompts.mode=prompted \
    prompts.n=1000 \
    prompts.fasta="$FASTA" \
    prompts.n_per=1000 \
    prompts.batch_size=4 \
    grpo.group_size=8 \
    grpo.max_new_tokens=96 \
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
    reward.terms="[$ENTROPY, $LENGTH, $RG]" \
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
    'hydra.sweep.subdir=${hydra.job.num}' \
    "$@"

echo "DONE -> $OUT"
