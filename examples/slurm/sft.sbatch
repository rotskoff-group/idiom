#!/bin/bash
# Supervised fine-tuning on 1 GPU: warm-start a released model and specialize it on one set.
# Submit from the repo root:  sbatch examples/slurm/sft.sbatch
#SBATCH --job-name=idiom-sft
#SBATCH --partition=gpu             # EDIT: your GPU partition
#SBATCH --account=your_account      # EDIT (or delete this line): your Slurm account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # 1 GPU -> 1 task (configs/sft.yaml uses devices: 1)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8           # DataLoader workers (data.num_workers)
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export IDIOM_OUT=/scratch/$USER/idiom-runs
export WANDB_MODE=offline           # EDIT: `wandb login` for live logging

# init_from takes a HF repo id, a released dir, or a .ckpt (e.g. a pretrain checkpoint from above).
srun idiom_train --config-name sft \
  init_from=jxliu2/idiom-300M \
  data.train_fasta=examples/example_data/protgps/nucleolus.fasta \
  data.prompted_prob=0.0            # whole-sequence IDRs (no flanks): train the de-novo form
