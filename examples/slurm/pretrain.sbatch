#!/bin/bash
# Pretrain IDiom on 8 GPUs (single node, DDP). Submit from the repo root:
#     sbatch examples/slurm/pretrain.sbatch
#SBATCH --job-name=idiom-pretrain
#SBATCH --partition=gpu             # EDIT: your GPU partition
#SBATCH --account=your_account      # EDIT (or delete this line): your Slurm account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8         # ONE TASK PER GPU — must equal trainer.devices
#SBATCH --gpus-per-node=8           # 8 GPUs (configs/pretrain.yaml uses devices: 8)
#SBATCH --cpus-per-task=8           # DataLoader workers per rank (data.num_workers)
#SBATCH --mem=0                     # all memory on the node
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out    # %x = job name, %j = job id

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"              # the repo root (where you ran sbatch)
source .venv/bin/activate          # the environment `uv sync` created

export IDIOM_OUT=/scratch/$USER/idiom-runs   # EDIT: checkpoints + Hydra dirs land here, not in the repo
export WANDB_MODE=offline                     # EDIT: `wandb login` (or set WANDB_API_KEY) for live logging

# Lightning reads the Slurm environment; srun launches one process per task (per GPU) and each
# attaches to one GPU. trainer.devices is GPUs-per-node (8, from the config) — keep it equal to
# --ntasks-per-node / --gpus-per-node above. For multi-node, raise --nodes and add
# `+trainer.num_nodes=$SLURM_NNODES` to the command below.
srun idiom_train \
  data.train_fasta=/path/to/train.fasta \
  data.val_fasta=/path/to/validation.fasta \
  model.n_layers=24 model.d_model=1024 model.n_heads=16
