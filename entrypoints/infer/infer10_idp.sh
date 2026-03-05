#!/bin/bash
#SBATCH --job-name=inf10idp
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm-%j.out

echo "===== BEGIN SLURM SCRIPT: $0 =====" # Save script into slurm out 
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SLURM SCRIPT: $0 ====="
echo; echo; echo; echo 

###
# Run inference for IDR FIM 
###

cd ..

# source /data2/scratch/group_scratch/idr_plm/idr-plm/.venv/bin/activate # H100
source /home/scratch/group_scratch/idr_plm/idr-plm/.venv/bin/activate # 4080 

# new FIM10 is version_3
CKPT_PATH="/home/scratch_mount/group_scratch/idr_plm/h100_rsync/idr_plm/2026-01-23_train_FIM/lightning_logs/version_3/checkpoints/best_model_step_246104.ckpt"

SHARD_PATH="/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/AFDB_IDR_90_FIM_512_splits/AFDB_IDR_90_FIM_512_splits_parts/precompute_shards/0001_file.h5"

PROMPT_PATH="prompts/idp_prompt_1e4x_array.pkl"

OUT_DIR="output/infer10_idp_1e4x"
mkdir -p $OUT_DIR

export PYTHONUNBUFFERED=1  
transformer_infer \
    "model=transformer" \
    "model.model=GeometricMolTransformer" \
    "model.model_args.unified_transformer_args.mha_args.mask_mode=causal" \
    "model.model_args.unified_transformer_args.n_layers=10" \
    "model.model_args.d_model=768" \
    "model.model_args.unified_transformer_args.mha_args.num_heads=12" \
    "model.model_args.unified_transformer_args.mha_layer_indices=[0,1,2,3,4,5,6,7,8,9]" \
    "training=transformer" \
    "training.lightning_model_args.optimizer_args.lr=4.0e-4" \
    "training.lightning_model_args.lr_scheduler=LinearWarmupCosineAnnealingLR" \
    "++training.lightning_model_args.lr_scheduler_args.warmup_epochs=3000" \
    "++training.lightning_model_args.lr_scheduler_args.max_epochs=250000" \
    "++training.lightning_model_args.lr_scheduler_args.eta_min=0.0" \
    "training.lightning_model_args.best_checkpoint_args.filename='best_model_{step}'" \
    "training.lightning_model_args.every_epoch_checkpoint_args.filename='restart_checkpoint'" \
    "training.lightning_model_args.every_epoch_checkpoint_args.every_n_epochs=null" \
    "training.lightning_model_args.every_epoch_checkpoint_args.every_n_train_steps=1000" \
    "training.lightning_model_args.every_epoch_checkpoint_args.save_top_k=1" \
    "training.training_mode=autoregressive" \
    "training.trainer_args.max_epochs=10000" \
    "training.trainer_args.max_steps=250000" \
    "training.trainer_args.devices=4" \
    "++training.trainer_args.val_check_interval=5000" \
    "++training.loss_fn_args.ignore_index=23" \
    "training.trainer_args.gradient_clip_val=null" \
    "training.trainer_args.gradient_clip_algorithm=null" \
    "+training.trainer_args.accumulate_grad_batches=1" \
    "training.resume_training_path=null" \
    "inference=transformer" \
    "inference.checkpoint_path=$CKPT_PATH" \
    "inference.savedir=$OUT_DIR" \
    "inference.inference_mode=autoregressive" \
    "inference.batch_size=100" \
    "inference.num_batches=100" \
    "inference.dataset_filename=$SHARD_PATH" \
    "inference.sampler_args.method=full" \
    "inference.sampler_args.sample_val=1" \
    "inference.sampler_args.temperature=1.0" \
    "++inference.addn_args.use_input_smiles=True" \
    "++inference.addn_args.smiles_path=$PROMPT_PATH"
