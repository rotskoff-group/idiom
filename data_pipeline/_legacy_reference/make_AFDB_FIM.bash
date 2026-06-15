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

# source /home/groups/ardunn/jxliu2/miniconda3/etc/profile.d/conda.sh
# conda activate plm
source /home/scratch/jxliu2/code_repos/idr-plm-figures/.venv/bin/activate

# PYTHON_FILE='/home/groups/ardunn/jxliu2/idr-analysis/src/data_preprocess/AFDB/make_AFDB_FIM.py'
PYTHON_FILE='/home/scratch/jxliu2/code_repos/idr-plm-figures/src/idr_plm_figures/data_preprocess/make_afdb_fim/make_AFDB_FIM.py'

# IN_FASTA="/oak/stanford/groups/rotskoff/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_50/AFDB_IDR_50_reps.fasta"
# IN_FASTA="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_reps.fasta"
IN_FASTA='/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/train_dedup_test_val.fasta'

# IN_H5_DIR="/oak/stanford/groups/rotskoff/AFDB/AFDB_v4_idr_alldata"
IN_H5_DIR="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata"

OUT_H5="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/AFDB_IDR_90_FIM_512_splits/AFDB_IDR_90_FIM_512_splits.h5"
# OUT_H5="/oak/stanford/groups/rotskoff/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_50/AFDB_IDR_50_FIM/AFDB_IDR_50_FIM_1019.h5"
# OUT_H5="/oak/stanford/groups/rotskoff/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_50/AFDB_IDR_50_FIM_512/AFDB_IDR_50_FIM_512.h5"

python3 -u "$PYTHON_FILE" \
        --fasta  "$IN_FASTA" \
        --h5_dir "$IN_H5_DIR" \
        --output "$OUT_H5"\