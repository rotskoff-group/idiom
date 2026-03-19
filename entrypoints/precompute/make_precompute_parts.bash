#!/bin/bash
#SBATCH --job-name=make_parts
#SBATCH --output=./slurm_out/slurm-%j.out

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

###
# Split one h5 file into N part_X_residues.h5 files and create matching
# *_targs.h5 files and the precompute_shards/ directory.
###

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/.venv/bin/activate"

IN_FILE="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_FIM_512/AFDB_IDR_90_FIM_512.h5"
OUT_DIR="$(dirname "${IN_FILE}")/$(basename "${IN_FILE}" .h5)_parts"

# mkdir -p slurm_out  # Run before submitting

make_precompute_parts "$IN_FILE" \
    --num-parts 500 \
    --out-dir "$OUT_DIR" \
    --seed 42
