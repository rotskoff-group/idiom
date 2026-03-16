#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --output=./slurm_out/slurm-%A_%a.out
#SBATCH --cpus-per-task=1
# #SBATCH --array=0-11%4

# Save script into slurm out 
echo "===== BEGIN SLURM SCRIPT: $0 =====" 
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SLURM SCRIPT: $0 ====="
echo; echo; echo; echo 

###
# Run RL post-training with the ProtGPS reward model 
###

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ENTRYPOINT_DIR}"

source "${REPO_ROOT}/.venv/bin/activate"

CKPT_PATH="${REPO_ROOT}/models/idr-plm/base/version_2/checkpoints/best_model_step_243022.ckpt"

SHARD_PATH="${REPO_ROOT}/data/shard/0001_file.h5"

DATASET_FILENAME="${REPO_ROOT}/data/rl_datasets/idp_dataset/idp_prompt_1e3x_grpo_dataset.h5"

PROTGPS_PARENT_DIR="${REPO_ROOT}/rewards/protgps"

# Training parameters 
LR=5e-6
BETA_KL=2e-2
REWARD_TARGET_VALUE=0.9
TARGET_LENGTH=100
LENGTH_REWARD_WEIGHT=1.0
LENGTH_REWARD_WIDTH=1
TARGET_ENTROPY=2.7
ENTROPY_REWARD_WEIGHT=1.0
ENTROPY_REWARD_WIDTH=0.2
BATCH_SIZE=4
ACCUMULATE_GRAD_BATCHES=2
GROUP_SIZE=8

COMPARTMENTS=(
    # "nuclear_speckle"
    # "p-body"
    # "pml-bdoy"
    # "post_synaptic_density"
    "stress_granule"
    # "chromosome"
    # "nucleolus"
    # "nuclear_pore_complex"
    # "cajal_body"
    # "rna_granule"
    # "cell_junction"
    # "transcriptional"
)

# Calculate job indexing based on SLURM_ARRAY_TASK_ID
NUM_COMPARTMENTS=${#COMPARTMENTS[@]}

# Indexing: compartment varies first
COMPARTMENT_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_COMPARTMENTS))

COMPARTMENT=${COMPARTMENTS[$COMPARTMENT_INDEX]}

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
    "data.dloader_args.batch_size=${BATCH_SIZE}"\
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
    "training.trainer_args.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"\
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
    "++training.lightning_model_args.group_size=${GROUP_SIZE}"\
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
