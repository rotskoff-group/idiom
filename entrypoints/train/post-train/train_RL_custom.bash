#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --output=./slurm_out/slurm-%A_%a.out
#SBATCH --cpus-per-task=1

###
# Run RL post-training with a generic reward function 
###

cd .. # Place this bash script into ./scripts of your working directory 

source /home/scratch/group_scratch/idr_plm/idr-plm/.venv/bin/activate # 4080 

CKPT_PATH="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/idr_plm/2026-01-26_train_FIM/lightning_logs/version_2/checkpoints/best_model_step_243022.ckpt"

SHARD_PATH="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_FIM_512/AFDB_IDR_90_FIM_512_parts/precompute_shards/0001_file.h5"

DATASET_FILENAME="dataset/idp_prompt_1e3x_grpo_dataset.h5"

PROTGPS_PARENT_DIR="/home/jxliu2/protgps"

# Training parameters 
LR=5e-6
BETA_KL_VALUES=(2e-2)
REWARD_TARGET_VALUES=(0.9)

TARGET_LENGTHS=(100)
LENGTH_REWARD_WEIGHTS=(1.0)
LENGTH_REWARD_WIDTHS=(1)

TARGET_ENTROPIES=(2.7)
ENTROPY_REWARD_WEIGHTS=(1.0)
ENTROPY_REWARD_WIDTHS=(0.2)

COMPARTMENTS=(
    # "nuclear_speckle"
    # "p-body"
    # "pml-bdoy"
    # "post_synaptic_density"
    # "stress_granule"
    # "chromosome"
    "nucleolus"
    "nuclear_pore_complex"
    "cajal_body"
    "rna_granule"
    "cell_junction"
    "transcriptional"
)

# Calculate job indexing based on SLURM_ARRAY_TASK_ID
NUM_COMPARTMENTS=${#COMPARTMENTS[@]}
NUM_TARGET_LENGTHS=${#TARGET_LENGTHS[@]}

# Indexing: compartment varies first, then target_length
COMPARTMENT_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_COMPARTMENTS))
TARGET_LENGTH_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_COMPARTMENTS))

BETA_KL=${BETA_KL_VALUES[0]}
LENGTH_REWARD_WEIGHT=${LENGTH_REWARD_WEIGHTS[0]}
TARGET_LENGTH=${TARGET_LENGTHS[$TARGET_LENGTH_INDEX]}
LENGTH_REWARD_WIDTH=${LENGTH_REWARD_WIDTHS[0]}
REWARD_TARGET_VALUE=${REWARD_TARGET_VALUES[0]}
ENTROPY_REWARD_WEIGHT=${ENTROPY_REWARD_WEIGHTS[0]}
TARGET_ENTROPY=${TARGET_ENTROPIES[0]}
COMPARTMENT=${COMPARTMENTS[$COMPARTMENT_INDEX]}
ENTROPY_REWARD_WIDTH=${ENTROPY_REWARD_WIDTHS[0]}

echo "======================================================================"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running with TARGET_LENGTH=${TARGET_LENGTH}, COMPARTMENT=${COMPARTMENT}"
echo "ENTROPY_REWARD_WEIGHT=${ENTROPY_REWARD_WEIGHT}, TARGET_ENTROPY=${TARGET_ENTROPY}"
echo "LR=${LR}"
echo "======================================================================"

# Set custom lightning log version name
export LIGHTNING_LOG_VERSION="version_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${COMPARTMENT}_target_len_${TARGET_LENGTH}_lr_${LR}"
echo "Lightning Log Version: ${LIGHTNING_LOG_VERSION}"

export PYTHONBREAKPOINT=ipdb.set_trace # for using breakpoint() 
export PYTHONUNBUFFERED=1

transformer_train \
    "data=transformer"\
    "data.dataset=TransformerOnlineDataset"\
    "data.collate_fn=transformer_online_collate_fn"\
    "data.dataset_filename=${DATASET_FILENAME}"\
    "data.dloader_args.batch_size=4"\
    "data.data_in_memory=False"\
    "model=transformer"\
    "model.model=GeometricMolTransformer"\
    "model.model_args.unified_transformer_args.mha_args.mask_mode=causal"\
    "model.model_args.unified_transformer_args.n_layers=12"\
    "model.model_args.d_model=896"\
    "model.model_args.unified_transformer_args.mha_args.num_heads=14"\
    "model.model_args.unified_transformer_args.mha_layer_indices=[0,1,2,3,4,5,6,7,8,9,10,11]"\
    "model.load_model=$CKPT_PATH"\
    "training=transformer"\
    "training.training_mode=grpo"\
    "training.trainer_args.max_epochs=100000"\
    "training.trainer_args.devices=1"\
    "training.trainer_args.strategy=ddp_find_unused_parameters_true"\
    "++training.trainer_args.limit_val_batches=0.0"\
    "++training.trainer_args.check_val_every_n_epoch=null"\
    "training.trainer_args.gradient_clip_val=null"\
    "training.trainer_args.gradient_clip_algorithm=null"\
    "training.trainer_args.max_steps=1500"\
    "training.trainer_args.accumulate_grad_batches=2"\
    "++training.trainer_args.log_every_n_steps=1"\
    "++training.loss_fn_args.ignore_index=23"\
    "++training.lightning_model_args.every_epoch_checkpoint_args.filename='step_{step:06d}'"\
    "++training.lightning_model_args.every_epoch_checkpoint_args.every_n_epochs=null"\
    "++training.lightning_model_args.every_epoch_checkpoint_args.every_n_train_steps=500"\
    "training.lightning_model_args.on_step=False"\
    "training.lightning_model_args.sync_dist=True"\
    "training.lightning_model_args.lr_scheduler=null"\
    "training.lightning_model_args.lr_scheduler_args=null"\
    "training.lightning_model_args.interval=null"\
    "training.lightning_model_args.optimizer_args.lr=${LR}"\
    "++training.lightning_model_args.sampler_args.method=full"\
    "++training.lightning_model_args.sampler_args.sample_val=1"\
    "++training.lightning_model_args.sampler_args.temperature=1"\
    "++training.lightning_model_args.sampler_args.token_limit=1000"\
    "++training.lightning_model_args.group_size=8"\
    "++training.lightning_model_args.epsilon_clip=0.2"\
    "++training.lightning_model_args.mu_grpo=1"\
    "++training.lightning_model_args.beta_kl=${BETA_KL}"\
    "++training.lightning_model_args.use_reward_shaping=True"\
    "++training.lightning_model_args.reward_target_value=${REWARD_TARGET_VALUE}"\
    "++training.lightning_model_args.reward_scale=1"\
    "++training.lightning_model_args.reward_function_name=compute_protgps_score"\
    "++training.lightning_model_args.protgps_target_compartment=${COMPARTMENT}"\
    "++training.lightning_model_args.protgps_aggregation=${COMPARTMENT}"\
    "++training.lightning_model_args.protgps_parent_dir=${PROTGPS_PARENT_DIR}"\
    "++training.lightning_model_args.normalize_advantage=True"\
    "++training.lightning_model_args.use_target_length=True"\
    "++training.lightning_model_args.target_length=${TARGET_LENGTH}"\
    "++training.lightning_model_args.length_reward_weight=${LENGTH_REWARD_WEIGHT}"\
    "++training.lightning_model_args.length_reward_width=${LENGTH_REWARD_WIDTH}"\
    "++training.lightning_model_args.use_target_entropy=True"\
    "++training.lightning_model_args.target_entropy=${TARGET_ENTROPY}"\
    "++training.lightning_model_args.entropy_reward_weight=${ENTROPY_REWARD_WEIGHT}"\
    "++training.lightning_model_args.entropy_reward_width=${ENTROPY_REWARD_WIDTH}"

