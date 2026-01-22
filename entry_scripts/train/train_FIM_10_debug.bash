#!/bin/bash
#SBATCH --job-name=FIM10
#SBATCH --time=7-00:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_out/slurm-%j.out 

###
# Run training for IDR FIM 
###

cd .. # Put this script under ./scripts in a working dir 

source /home/scratch/group_scratch/idr_plm/idr-plm/.venv/bin/activate 

export PYTHONBREAKPOINT=ipdb.set_trace # for using breakpoint() 
export PYTHONUNBUFFERED=1

transformer_train "data=transformer"\
	"data.dataset=TransformerShardedAutoregDataset"\
	"data.collate_fn=transformer_sharded_autoreg_collate_fn"\
	"data.dataset_filename=/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/AFDB_IDR_90_FIM_512_splits/AFDB_IDR_90_FIM_512_splits_parts/precompute_shards"\
	"data.splits=/home/scratch_mount/group_scratch/idr_plm/sherlock_rsync/AFDB/AFDB_v4_idr_alldata/clustering/AFDB_IDR_90/AFDB_IDR_90_splits/AFDB_IDR_90_FIM_512_splits/fim_split_indices.npy"\
	"data.dataset_split_args.train=0.995" \
	"data.dataset_split_args.val=0.005" \
	"data.dloader_args.batch_size=256" \
	"data.dloader_args.num_workers=4"\
	"data.data_in_memory=False"\
	"model=transformer"\
	"model.model=GeometricMolTransformer"\
	"model.model_args.unified_transformer_args.mha_args.mask_mode=causal"\
	"model.model_args.unified_transformer_args.n_layers=10" \
	"model.model_args.d_model=768" \
	"model.model_args.unified_transformer_args.mha_args.num_heads=12" \
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
	"training.trainer_args.devices=2"\
	"++training.trainer_args.val_check_interval=5000" \
	"++training.loss_fn_args.ignore_index=23"\
	"training.trainer_args.gradient_clip_val=null"\
	"training.trainer_args.gradient_clip_algorithm=null"\
	"+training.trainer_args.accumulate_grad_batches=2"\
	"training.resume_training_path=null"
	

	# IGNORE_INDEX will be 20 for causal (no FIM sentinels), 23 for FIM 
	# "training.lightning_model_args.lr_scheduler=null" \
