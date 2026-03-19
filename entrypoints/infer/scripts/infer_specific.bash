#!/bin/bash
#SBATCH --job-name=gen_idr
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./slurm_out/slurm-%j.out 

echo "===== BEGIN SLURM SCRIPT: $0 =====" # Save script into slurm out 
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SLURM SCRIPT: $0 ====="
echo; echo; echo; echo 

###
# Generated prompted IDRs
###

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR" && git rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
ENTRYPOINT_DIR="${REPO_ROOT}/entrypoints/infer"
cd "${ENTRYPOINT_DIR}"

source "${REPO_ROOT}/.venv/bin/activate"

# new FIM10 is version_3
CKPT_PATH="${REPO_ROOT}/models/idr-plm/base/version_2/checkpoints/best_model_step_243022.ckpt"

SHARD_PATH="${REPO_ROOT}/models/data/shard/0001_file.h5"

PROMPT_PATH="${REPO_ROOT}/models/data/prompts/p06748_prompt_1e5x_array.pkl"

OUT_DIR="${REPO_ROOT}/entrypoints/infer/output"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1  
transformer_infer \
    "model=transformer" \
    "model.model=GeometricMolTransformer" \
    "model.model_args.unified_transformer_args.mha_args.mask_mode=causal" \
    "model.model_args.unified_transformer_args.n_layers=12" \
    "model.model_args.d_model=896" \
    "model.model_args.unified_transformer_args.mha_args.num_heads=14" \
    "model.model_args.unified_transformer_args.mha_layer_indices=[0,1,2,3,4,5,6,7,8,9,10,11]" \
    "training=transformer" \
    "training.lightning_model_args.optimizer_args.lr=4.0e-4" \
    "training.lightning_model_args.lr_scheduler=LinearWarmupCosineAnnealingLR" \
    "++training.lightning_model_args.lr_scheduler_args.warmup_epochs=3000" \
    "++training.lightning_model_args.lr_scheduler_args.max_epochs=250000" \
    "++training.lightning_model_args.lr_scheduler_args.eta_min=4.0e-5" \
    "training.lightning_model_args.best_checkpoint_args.filename='best_model_{step}'" \
    "training.lightning_model_args.every_epoch_checkpoint_args.filename='restart_checkpoint'" \
    "training.lightning_model_args.every_epoch_checkpoint_args.every_n_epochs=null" \
    "training.lightning_model_args.every_epoch_checkpoint_args.every_n_train_steps=1000" \
    "training.lightning_model_args.every_epoch_checkpoint_args.save_top_k=1" \
    "training.training_mode=autoregressive" \
    "training.trainer_args.max_epochs=10000" \
    "training.trainer_args.max_steps=250000" \
    "training.trainer_args.devices=8" \
    "++training.trainer_args.val_check_interval=25000" \
    "++training.loss_fn_args.ignore_index=23" \
    "training.trainer_args.gradient_clip_val=null" \
    "training.trainer_args.gradient_clip_algorithm=null" \
    "training.trainer_args.accumulate_grad_batches=1" \
    "training.resume_training_path=null" \
    "inference=transformer" \
    "inference.checkpoint_path=$CKPT_PATH" \
    "inference.savedir=$OUT_DIR" \
    "inference.inference_mode=autoregressive" \
    "inference.batch_size=100" \
    "inference.use_multi_gpu=True" \
    "inference.dataset_filename=$SHARD_PATH" \
    "inference.sampler_args.method=full" \
    "inference.sampler_args.sample_val=1" \
    "inference.sampler_args.temperature=1.0" \
    "++inference.addn_args.use_input_smiles=True" \
    "++inference.addn_args.smiles_path=$PROMPT_PATH"
