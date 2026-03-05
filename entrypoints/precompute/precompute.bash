#!/bin/bash
#SBATCH --job-name=precomp
#SBATCH --output=outfiles/precompute_%A_%a.out
#SBATCH --error=outfiles/precompute_%A_%a.err

#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --array=1-500%16 # These values should match filename indices 

###
# Run precompute 
###

source /home/scratch/group_scratch/idr_plm/idr-plm/.venv/bin/activate 

TASK_ID=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})

# mkdir -p outfiles # Need to run these before running the script 
# mkdir -p outputs 

# Need to first create the precompute_shards subdirectory (currently done in make_targs.py) 
transformer_precompute "precompute=smiles"\
    "precompute.smiles_file=part_${SLURM_ARRAY_TASK_ID}_idrs.h5"\
    "precompute.target_file=part_${SLURM_ARRAY_TASK_ID}_targs.h5"\
    "precompute.output_file=precompute_shards/${TASK_ID}_file.h5"\
    "precompute.input_generator_addn_args.apply_start=true"\
    "precompute.input_generator_addn_args.apply_stop=false"\
    "precompute.target_generator_addn_args.apply_start=false"\
    "precompute.target_generator_addn_args.apply_stop=true"\
    "precompute.tokenizer=CharTokenizer"

