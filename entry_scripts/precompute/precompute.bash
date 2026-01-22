#!/bin/bash
#SBATCH --job-name=precomp
#SBATCH --output=outfiles/precompute_%A_%a.out
#SBATCH --error=outfiles/precompute_%A_%a.err

#SBATCH --partition=rotskoff
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --array=1-500%16 # These values should match filename indices 
# #SBATCH --nodelist=sh03-11n15 # A40 node 
# #SBATCH --nodelist=sh04-12n01 # H100 node 

###
# Run precompute 
###

source /home/groups/ardunn/jxliu2/MoLE/.venv/bin/activate

TASK_ID=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})

# Need to first create the precompute_shards subdirectory (currently done in make_targs.py) 
mgpt_transformer_precompute "precompute=smiles"\
    "precompute.smiles_file=part_${SLURM_ARRAY_TASK_ID}_idrs.h5"\
    "precompute.target_file=part_${SLURM_ARRAY_TASK_ID}_targs.h5"\
    "precompute.output_file=precompute_shards/${TASK_ID}_file.h5"\
    "precompute.input_generator_addn_args.apply_start=true"\
    "precompute.input_generator_addn_args.apply_stop=false"\
    "precompute.target_generator_addn_args.apply_start=false"\
    "precompute.target_generator_addn_args.apply_stop=true"\
    "precompute.tokenizer=CharTokenizer"

