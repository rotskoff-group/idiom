#!/bin/bash
#SBATCH --job-name=idiom-sae
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

###
# Train a top-k SAE, streaming activations from a frozen IDiom (no activation cache on disk).
# layer must be < n_layers (18 matches the released idiomsae-300M-L18-k32); expansion_factor=32
# gives 32 x d_model latents. Resume by hand with resume_from=<ckpt>.
###

REPO=/path/to/idiom                     # EDIT
OUT=/path/to/runs/sae                   # EDIT: keep runs out of the repo
FASTA=/path/to/records.fasta            # EDIT

source /path/to/venv/bin/activate       # EDIT
cd "$REPO"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then mkdir -p slurm_out; fi   # #SBATCH --output writes here
export WANDB_MODE=offline               # `wandb login` and set online for live logging

idiom_sae \
    seed=0 \
    device=auto \
    model_ckpt=jxliu2/idiom-300M \
    resume_from=null \
    layer=18 \
    region=all \
    data.fasta="$FASTA" \
    data.prompted_prob=0.5 \
    data.record_batch_size=16 \
    sae_batch_size=4096 \
    buffer_size=262144 \
    init_b_dec_from_mean=true \
    sae.k=32 \
    sae.expansion_factor=32 \
    sae.activation=topk \
    sae.multi_topk=false \
    sae.auxk_alpha=0.03125 \
    sae.dead_feature_tokens=10000000 \
    sae.warmup_steps=1000 \
    trainer.max_steps=100000 \
    trainer.accelerator=auto \
    trainer.devices=1 \
    out_dir="$OUT" \
    hydra.run.dir="$OUT/hydra"

echo "DONE -> $OUT"
