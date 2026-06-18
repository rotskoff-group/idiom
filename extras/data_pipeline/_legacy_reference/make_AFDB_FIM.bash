#!/bin/bash
#SBATCH --job-name=fim
#SBATCH --output=./slurm_out/slurm-%j.out

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB 

###
# Create a FIM dataset from a FASTA file containing AFDB clustered to e.g. 50 
###

source /path/to/venv/bin/activate

PYTHON_FILE='./make_AFDB_FIM.py'

IN_FASTA='/path/to/AFDB/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/train_dedup_test_val.fasta'

IN_H5_DIR="/path/to/AFDB/AFDB_v4_idr_alldata"

OUT_H5="/path/to/AFDB/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/AFDB_IDR_90_FIM_512_splits.h5"

python3 -u "$PYTHON_FILE" \
        --fasta  "$IN_FASTA" \
        --h5_dir "$IN_H5_DIR" \
        --output "$OUT_H5"\