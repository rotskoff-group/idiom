#!/bin/bash
#SBATCH --job-name=idiom-sae
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu             # EDIT: your GPU partition
# #SBATCH --account=your_account    # EDIT: uncomment if your site requires an account
#SBATCH --output=./slurm_out/slurm-%j.out   # sbatch only; run `mkdir -p slurm_out` first

echo "===== BEGIN SCRIPT: $0 ====="
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SCRIPT: $0 ====="
echo; echo

set -euo pipefail

###
# Train a top-k SAE on 1 GPU, streaming activations from a FROZEN IDiom (no activation cache on disk).
# model_ckpt is the frozen base: a HF repo id, a released dir, or a .ckpt. layer must be < n_layers
# (18 matches the released idiomsae-300M-L18-k32). expansion_factor=32 -> 32*d_model latents.
# Resume by hand: pass resume_from=<path to a Lightning ckpt> (no fixed last.ckpt here).
###

# Repo root: where sbatch was submitted from, or this script's own location under bash.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${IDIOM_OUT:-$HOME/idiom-runs}/sae"          # EDIT: keep runs on scratch, not in the repo

unset PYTHONPATH PYTHONHOME
# The clone's own venv if `uv sync` made one; otherwise whatever environment idiom is installed in
# (a `pip install git+...` into your own env needs no activation here).
if [[ -f "$REPO/.venv/bin/activate" ]]; then source "$REPO/.venv/bin/activate"; fi
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=${WANDB_MODE:-offline}           # EDIT: `wandb login` and set online for live logging
echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

idiom_sae \
    seed=0 \
    device=auto \
    model_ckpt=jxliu2/idiom-300M \
    resume_from=null \
    layer=18 \
    region=all \
    data.fasta=/path/to/records.fasta \
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
