#!/bin/bash
#SBATCH --job-name=precomp
#SBATCH --output=./slurm_out/slurm-%j.out 
# #SBATCH --output=outfiles/precompute_%A_%a.out
# #SBATCH --error=outfiles/precompute_%A_%a.err

#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --array=1-500%16 # These values should match --num-parts below

###
# Combined: split input into parts (task 1 only), then run precompute on each part
###

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/slurm_out"

source "${REPO_ROOT}/.venv/bin/activate"

IN_FILE="${REPO_ROOT}/datasets/idr_datasets/training_sequences/AFDB_IDR_90_FIM_512.h5"
OUT_DIR="$(dirname "${IN_FILE}")/$(basename "${IN_FILE}" .h5)_parts"
NUM_PARTS=500

# mkdir -p outfiles # Need to run these before running the script

# Task 1 splits the input file and creates precompute_shards/
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    make_precompute_parts "$IN_FILE" \
        --num-parts "$NUM_PARTS" \
        --out-dir "$OUT_DIR" \
        --seed 42
fi

# All tasks wait until precompute_shards/ exists (created at end of make_precompute_parts)
while [ ! -d "${OUT_DIR}/precompute_shards" ]; do
    sleep 10
done

TASK_ID=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})

transformer_precompute "precompute=residues" \
    "precompute.residues_file=${OUT_DIR}/part_${SLURM_ARRAY_TASK_ID}_residues.h5" \
    "precompute.target_file=${OUT_DIR}/part_${SLURM_ARRAY_TASK_ID}_targs.h5" \
    "precompute.output_file=${OUT_DIR}/precompute_shards/${TASK_ID}_file.h5" \
    "precompute.input_generator_addn_args.apply_start=true" \
    "precompute.input_generator_addn_args.apply_stop=false" \
    "precompute.target_generator_addn_args.apply_start=false" \
    "precompute.target_generator_addn_args.apply_stop=true" \
    "precompute.tokenizer=CharTokenizer"
