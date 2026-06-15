#!/bin/bash
#SBATCH --job-name=mmseqs2
#SBATCH --output=./slurm_out/slurm-%j.out
#SBATCH --error=./slurm_out/slurm-%j.err

#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=8GB 

###
# Run MMseqs2 to cluster sequences from AFDB 
# alldata_seqs.fasta contains all full-length sequences for AFDB sequences that contain an IDR 
###

PREFIX=AFDB_IDR_90 # prefix for MMseqs files

FASTA=/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/alldata_seqs.fasta # 110 M full length sequences
# This clustering will actually remove IDR instances for which there are >1 IDR per protein 
SCRATCH=/home/scratch_mount/group_scratch/idr_plm/mmseqs2_scratch/mmseqs2_${SLURM_JOB_ID}
THREADS=$SLURM_CPUS_PER_TASK
RESULT_DIR=/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/${PREFIX}
mkdir -p "$RESULT_DIR"
cd "$RESULT_DIR"

source /home/scratch/jxliu2/miniconda3/etc/profile.d/conda.sh
conda activate mmseqs2
# source /home/scratch/jxliu2/code_repos/idr-plm-figures/.venv/bin/activate

mkdir -p "$SCRATCH"

echo "Step 1 convert FASTA to DB"
mmseqs createdb "$FASTA" "${PREFIX}_db" -v 3 

echo "Step 2 Linclust"
mmseqs linclust "${PREFIX}_db"  "${PREFIX}_clu90"  "$SCRATCH" \
        --threads "$THREADS" \
        --min-seq-id 0.9 \
        --cov-mode 0 -c 0.8 \
        --cluster-mode 2 \
        -v 3 

echo "Step 3 Extract representatives"
mmseqs createsubdb  "${PREFIX}_clu90"  "${PREFIX}_db"  "${PREFIX}_reps" -v 3
mmseqs convert2fasta "${PREFIX}_reps"  "${PREFIX}_reps.fasta" -v 3 

echo "Step 4 Write cluster mapping"
mmseqs createtsv "${PREFIX}_db" "${PREFIX}_db" \
                 "${PREFIX}_clu90"  "${PREFIX}_mapping.tsv" -v 3 

echo "All done. Representative FASTA: ${PREFIX}_reps.fasta"
echo "Cluster map: ${PREFIX}_mapping.tsv"

