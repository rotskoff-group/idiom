#!/bin/bash
# GRPO / RL post-training on 1 GPU. Submit from the repo root:  sbatch examples/slurm/grpo.sbatch
#SBATCH --job-name=idiom-grpo
#SBATCH --partition=gpu             # EDIT: your GPU partition
#SBATCH --account=your_account      # EDIT (or delete this line): your Slurm account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 GPU -> 1 task (configs/grpo.yaml uses devices: 1)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export IDIOM_OUT=/scratch/$USER/idiom-runs
export WANDB_MODE=offline
# export UV_CACHE_DIR=/scratch/$USER/uv-cache   # needed if a reward uses an on-demand uv env (sparrow)

# RL toward an SAE feature signature (no third-party reward model; runs straight after `uv sync`).
srun idiom_grpo \
  init_from=jxliu2/idiom-300M \
  reward.rl_sae.enabled=true reward.rl_sae.signature=nucleolus

# --- variant: an external reward model (sparrow) in its own env (uncomment, and set UV_CACHE_DIR) ---
# srun idiom_grpo init_from=jxliu2/idiom-300M \
#   '+reward.external=[{enabled: true, weight: 0.5, target: 25, width: 0.2, cmd: "uv run --isolated --no-project --with '"'"'sparrow @ git+https://github.com/idptools/sparrow.git'"'"' python rewards/external_scorers/sparrow.py --property radius_of_gyration"}]'
