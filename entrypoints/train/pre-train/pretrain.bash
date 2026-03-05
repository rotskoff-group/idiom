#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --time=7-00:00:00
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16GB
#SBATCH --output=./slurm_out/slurm-%j.out 

echo "===== BEGIN SLURM SCRIPT: $0 =====" # Save script into slurm out 
sed -e 's/^/    /' "${BASH_SOURCE[0]}"
echo "===== END   SLURM SCRIPT: $0 ====="
echo; echo; echo; echo 

###
# Run pre-training 
###

cd .. # Put this script under ./scripts in a day's working dir 

source /home/groups/ardunn/jxliu2/idr-plm/.venv/bin/activate # sherlock 

export PYTHONBREAKPOINT=ipdb.set_trace # for using breakpoint() 
export PYTHONUNBUFFERED=1

# Dataset path
DATASET_FILENAME="/scratch/groups/ardunn/jxliu2/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90_FIM_512/AFDB_IDR_90_FIM_512_parts/precompute_shards" # sherlock

echo 'Start training..' 

transformer_train \
	"data=transformer"\
	"data.dataset=TransformerShardedAutoregDataset"\
	"data.collate_fn=transformer_sharded_autoreg_collate_fn"\
	"data.dataset_filename=${DATASET_FILENAME}"\
	"data.splits=null"\
	"data.dataset_split_args.train=0.99"\
	"data.dataset_split_args.val=0.005"\
	"data.dataset_split_args.test=0.005"\
	"data.dloader_args.batch_size=128" \
	"training.trainer_args.devices=8"\
	"training.trainer_args.accumulate_grad_batches=1"\
	"data.data_in_memory=False"\
	"data.dloader_args.num_workers=8"\
	"model=transformer"\
	"model.model=GeometricMolTransformer"\
	"model.model_args.unified_transformer_args.mha_args.mask_mode=causal"\
	"model.model_args.unified_transformer_args.n_layers=12" \
	"model.model_args.unified_transformer_args.mha_layer_indices=[0,1,2,3,4,5,6,7,8,9,10,11]"\
	"model.model_args.d_model=896" \
	"model.model_args.unified_transformer_args.mha_args.num_heads=14" \
	"training=transformer"\
	"training.lightning_model_args.optimizer_args.lr=4.0e-4" \
	"training.lightning_model_args.lr_scheduler=LinearWarmupCosineAnnealingLR" \
	"++training.lightning_model_args.lr_scheduler_args.warmup_epochs=3000" \
	"++training.lightning_model_args.lr_scheduler_args.max_epochs=250000" \
	"++training.lightning_model_args.lr_scheduler_args.eta_min=4.0e-5" \
	"training.lightning_model_args.best_checkpoint_args.filename='best_model_{step}'"\
	"training.lightning_model_args.every_epoch_checkpoint_args.filename='restart_checkpoint'"\
	"training.lightning_model_args.every_epoch_checkpoint_args.every_n_epochs=null"\
	"training.lightning_model_args.every_epoch_checkpoint_args.every_n_train_steps=1000"\
	"training.lightning_model_args.every_epoch_checkpoint_args.save_top_k=1"\
	"training.training_mode=autoregressive"\
	"training.trainer_args.max_epochs=10000"\
	"training.trainer_args.max_steps=250000" \
	"++training.trainer_args.val_check_interval=25000" \
	"++training.loss_fn_args.ignore_index=23"\
	"training.trainer_args.gradient_clip_val=null"\
	"training.trainer_args.gradient_clip_algorithm=null"\
	"training.resume_training_path=null"
	
	# IGNORE_INDEX is 23 for FIM IDR-PLM  