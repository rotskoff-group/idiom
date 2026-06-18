#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --output=%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
# Curation stage 3 — filter the curated master h5 -> record FASTA.
# Applies length (<= max_len-4 = 1020) + fully-low-pLDDT (aggressive no-folded-segment) filters,
# writing `>{base}_IDR_{x}-{y}` + full_seq. CPU-only but heavy I/O: one linear pass over the
# ~125G, 73M-row master (the full_avg_plddt vlen arrays are the bulk), so it is a real job.
#
#   sbatch data_pipeline/filter_length_plddt.bash
#   # or interactively:  srun -c 4 --mem 16GB -t 12:00:00 bash data_pipeline/filter_length_plddt.bash
set -euo pipefail

REPO=/data2/scratch/jxliu2/idiom
source "$REPO/.venv/bin/activate"
cd "$REPO"

# --- params (every value spelled out / overridable) ---
DATA=/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data
AFDB="$DATA/pretraining/AFDB"
H5="${H5:-$AFDB/clustering_90/AFDB_IDR_90_alldata.h5}"
MAX_LEN="${MAX_LEN:-1024}"   # protein cap is MAX_LEN-4
CAP=$((MAX_LEN - 4))
# name encodes source (AFDB_IDR_90 master) + filters (len cap + fully-low-pLDDT proteins removed)
OUT="${OUT:-$AFDB/intermediate/AFDB_IDR_90_len${CAP}_rm_full_low_plddt.fasta}"

mkdir -p "$(dirname "$OUT")"
echo "in : $H5"
echo "out: $OUT   (max_len=$MAX_LEN -> cap $CAP)"

python -m data_pipeline.filter_length_plddt \
    --h5      "$H5" \
    --out     "$OUT" \
    --max-len "$MAX_LEN"

echo "Done. Next: stage 4 DisProt dedup -> bash data_pipeline/dedup.bash \"$OUT\" \"$AFDB/intermediate/dedup\""
