#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=./slurm_out/slurm-%j.out

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

###
# Split one h5 file into multiple part h5's, e.g. 500 or 1000 
###

source /home/scratch/group_scratch/idr_plm/idr-plm-figures/.venv/bin/activate

PY_SCRIPT="./split_parts.py"

# OUT_DIR="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/AFDB_IDR_90_FIM_512_splits/AFDB_IDR_90_FIM_512_splits_parts" # Clust 90 and dedup 
OUT_DIR="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_FIM_512/AFDB_IDR_90_FIM_512_parts" # Just clust 90 

# IN_FILE="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/AFDB_IDR_90_FIM_512_splits/AFDB_IDR_90_FIM_512_splits_idrs.h5"
IN_FILE="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_FIM_512/AFDB_IDR_90_FIM_512.h5"

python -u "$PY_SCRIPT" "$IN_FILE" \
    --num-parts 500 \
    --out-dir "$OUT_DIR" \
