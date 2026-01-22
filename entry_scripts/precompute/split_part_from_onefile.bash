#!/bin/bash
#SBATCH --job-name=splt1024
#SBATCH --output=/oak/stanford/groups/rotskoff/AFDB/logs/split_part/split_part_%j.out
#SBATCH --error=/oak/stanford/groups/rotskoff/AFDB/logs/split_part/split_part_%j.err

#SBATCH --partition=rotskoff
#SBATCH --time=0-05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# # #SBATCH --nodelist=sh03-11n15 # A40 node 
#SBATCH --mem-per-cpu=32G
#SBATCH --nodelist=sh04-12n01 # H100 node 

###
# Split one h5 file into multiple part h5's, e.g. 500 or 1000 
###

PY_SCRIPT="/home/groups/ardunn/jxliu2/idr-analysis/src/data_preprocess/AFDB/split_part_from_onefile.py"

OUT_DIR="/oak/stanford/groups/rotskoff/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_50/AFDB_IDR_50_FIM_1024/AFDB_IDR_50_FIM_1024_parts/"

IN_FILE="/oak/stanford/groups/rotskoff/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_50/AFDB_IDR_50_FIM_1024/AFDB_IDR_50_FIM_1024.h5"

python -u "$PY_SCRIPT" "$IN_FILE" \
    --num-parts 500 \
    --out-dir "$OUT_DIR" \
