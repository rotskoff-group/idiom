#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --time=1-00:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=./slurm_out/slurm-%j.out

# You can run this script using 'sbatch extract_activations.bash' or 'bash extract_activations.bash'

# Create slurm_out if using SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    mkdir -p ./slurm_out
fi

echo "===== BEGIN SLURM SCRIPT: $0 =====" # Save script into slurm out
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SLURM SCRIPT: $0 ====="
echo; echo; echo; echo

###
# Extract hidden-state activations from the pre-trained base IDiom model.
# This example runs on the two-protein FASTA used by the IDR generation demo
# Point DATA_PATH at your own data to extract activations at scale.
###

# Determine repository root when using either SLURM or bash to run
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR" && git rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

echo "Repo root: " ${REPO_ROOT}

source "${REPO_ROOT}/.venv/bin/activate"

# SET YOUR DESIRED MODEL CHECKPOINT PATH HERE:
CKPT_PATH="${REPO_ROOT}/models/idiom/base/version_2/checkpoints/best_model_step_243022.ckpt" # Pretrained base model

# Alternatively, activations can be extracted from any of the post-trained models:
# CKPT_PATH="${REPO_ROOT}/models/idiom/post_trained/protgps_reward/version_59392_0_nucleolus_target_len_100_lr_5e-6/checkpoints/step_step_001500.ckpt"      # nucleolus
# CKPT_PATH="${REPO_ROOT}/models/idiom/post_trained/protgps_reward/version_59380_5_chromosome_target_len_100_lr_5e-6/checkpoints/step_step_001500.ckpt"     # chromosome
# CKPT_PATH="${REPO_ROOT}/models/idiom/post_trained/protgps_reward/version_59380_1_p-body_target_len_100_lr_5e-6/checkpoints/step_step_001500.ckpt"         # p-body
# CKPT_PATH="${REPO_ROOT}/models/idiom/post_trained/protgps_reward/version_59380_4_stress_granule_target_len_100_lr_5e-6/checkpoints/step_step_001500.ckpt" # stress_granule

# Input sequences. This may be either:
#   - a FASTA with "_IDR_x-y" headers (tokenized into FIM format 1{prefix}3{suffix}2{IDR}
#     automatically), or
#   - an .h5 file with a "residues" field of already-FIM-formatted residue strings.
# DATA_PATH="${REPO_ROOT}/entrypoints/generate/scripts/example_sequences.fasta"
DATA_PATH="${REPO_ROOT}/datasets/idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_small.h5" # 100k sequences here
# DATA_PATH="/home/scratch/jxliu2/code_repos/idiom/datasets/idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_small.h5" # 100k sequences here 

OUT_DIR="${REPO_ROOT}/entrypoints/extract_activations/output"
OUTPUT_DIR="${OUT_DIR}/activations_layer_8"
NSHARDS=64
mkdir -p "${OUTPUT_DIR}"

echo "Starting activation extraction"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HDF5_USE_FILE_LOCKING=FALSE

transformer_extract \
    "model=transformer" \
    "training=transformer" \
    "extract=transformer" \
    "++extract.checkpoint_path=$CKPT_PATH" \
    "++extract.dataset_filename=$DATA_PATH" \
    "++extract.output_dir=$OUTPUT_DIR" \
    "++extract.layers=[8]" \
    "++extract.batch_size=256" \
    "++extract.save_dtype=float32" \
    "++extract.num_precompute_workers=null" \
    "++extract.num_shards=$NSHARDS" \
    "++extract.max_sequences=null"

# "++extract.layers=[0,1,2,3,4,5,6,7,8,9,10,11]" \

echo
echo "Completed activation extraction"
echo
