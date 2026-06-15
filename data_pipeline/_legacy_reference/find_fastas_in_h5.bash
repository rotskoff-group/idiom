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

# source /home/groups/ardunn/jxliu2/miniconda3/etc/profile.d/conda.sh
# conda activate plm
source /home/scratch/jxliu2/code_repos/idr-plm-figures/.venv/bin/activate

PYTHON_FILE='/home/scratch/jxliu2/code_repos/idr-plm-figures/src/idr_plm_figures/data_preprocess/clusters_to_h5/find_fastas_in_h5.py'

IN_FASTA="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_reps.fasta"

IN_H5_DIR="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata"

OUT_H5="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90_alldata.h5"

python3 -u "$PYTHON_FILE" \
        --fasta  "$IN_FASTA" \
        --h5_dir "$IN_H5_DIR" \
        --output "$OUT_H5"\

# python3 extract_h5_records_by_sequence.py \
#     --fasta   path/to/your.fasta \
#     --h5_dir  path/to/shards/ \
#     --output  matched_records.h5
