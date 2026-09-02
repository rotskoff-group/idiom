#!/bin/bash
# Train a top-k SAE on 1 GPU (streaming activations from a frozen IDiom). Submit from the repo root:
#     sbatch examples/slurm/sae.sbatch
#SBATCH --job-name=idiom-sae
#SBATCH --partition=gpu             # EDIT: your GPU partition
#SBATCH --account=your_account      # EDIT (or delete this line): your Slurm account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 GPU -> 1 task (configs/sae.yaml uses devices: 1)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export IDIOM_OUT=/scratch/$USER/idiom-runs
export WANDB_MODE=offline

# model_ckpt is the frozen base: a HF repo id, a released dir, or a .ckpt. layer must be < n_layers.
srun idiom_sae \
  model_ckpt=jxliu2/idiom-300M \
  data.fasta=/path/to/records.fasta \
  layer=18 sae.k=32
