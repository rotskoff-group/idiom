#!/bin/bash
#SBATCH --job-name=fasta_h5
#SBATCH --output=./slurm_out/slurm-%j.out
#SBATCH --error=./slurm_out/slurm-%j.err

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32GB 

###
# Find FASTA sequences output from MMseqs2 within alldata h5 set, and write to new h5 file 
###

source /path/to/venv/bin/activate

PYTHON_FILE='./find_fastas_in_h5.py'

IN_FASTA="/path/to/AFDB/clustering/AFDB_IDR_90/AFDB_IDR_90_reps.fasta"

IN_H5_DIR="/path/to/AFDB/AFDB_v4_idr_alldata"

OUT_H5="/path/to/AFDB/clustering/AFDB_IDR_90_alldata.h5"

python3 -u "$PYTHON_FILE" \
        --fasta  "$IN_FASTA" \
        --h5_dir "$IN_H5_DIR" \
        --output "$OUT_H5"\

# python3 extract_h5_records_by_sequence.py \
#     --fasta   path/to/your.fasta \
#     --h5_dir  path/to/shards/ \
#     --output  matched_records.h5
