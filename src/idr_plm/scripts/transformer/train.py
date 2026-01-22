import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from lightning.pytorch.strategies import FSDPStrategy
from clm.utils.data_utils import get_hdf5_fn, split_data_subsets
from clm.utils.misc import seed_worker
from mole.nn.transformer.module import LightningModel
from mole.nn.transformer import dataset as DSET
from mole.utils.token import aggregate_tokens_hdf5
import os
import h5py


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="training")
def main(cfg) -> None:
    model_args = cfg["model"]
    training_args = cfg["training"]
    data_args = cfg["data"]

    L.seed_everything(**training_args["seed_args"])

    get_hdf5 = get_hdf5_fn(data_args["dataset_filename"])

    dset_base = getattr(DSET, data_args["dataset"])
    collate_fn = getattr(DSET, data_args["collate_fn"])

    dataset = dset_base(
        get_hdf5,
        data_in_memory=data_args["data_in_memory"],
        **data_args["dataset_args"],
    )

    if os.path.isdir(data_args["dataset_filename"]):
        all_h5_files = [
            os.path.join(data_args["dataset_filename"], f)
            for f in os.listdir(data_args["dataset_filename"])
            if f.endswith(".h5")
        ]
        if len(all_h5_files) == 0:
            raise ValueError("No h5 files found in the directory")
        inFile = h5py.File(all_h5_files[0], "r")
    else:
        inFile = h5py.File(data_args["dataset_filename"], "r")
    token_dict = aggregate_tokens_hdf5(inFile)
    del inFile

    print("Setting up dataloaders...")
    train_set, val_set, test_set = split_data_subsets(
        dataset,
        data_args["splits"],
        data_args["dataset_split_args"]["train"],
        data_args["dataset_split_args"]["val"],
        data_args["dataset_split_args"]["test"],
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_set,
        worker_init_fn=seed_worker,
        generator=g,
        **data_args["dloader_args"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        worker_init_fn=seed_worker,
        generator=g,
        **data_args["dloader_args"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    # test_loader = DataLoader(test_set, worker_init_fn=seed_worker,
    #                          generator=g, **data_args['dloader_args'], shuffle=False, collate_fn=collate_fn)

    every_epoch_checkpoint_callback = ModelCheckpoint(
        **training_args["lightning_model_args"]["every_epoch_checkpoint_args"]
    )
    best_checkpoint_callback = ModelCheckpoint(
        **training_args["lightning_model_args"]["best_checkpoint_args"]
    )
    lr_monitor = LearningRateMonitor(
        **training_args["lightning_model_args"]["lr_monitor_args"]
    )

    custom_version = os.environ.get("LIGHTNING_LOG_VERSION", None)
    logger = TensorBoardLogger(save_dir="./", version=custom_version)
    lightning_model = LightningModel(
        model_args=model_args, token_info=token_dict, training_args=training_args
    )

    if training_args["trainer_args"]["devices"] != 0:
        if training_args["trainer_args"]["strategy"] == "fsdp":
            training_args["trainer_args"]["strategy"] = FSDPStrategy(
                activation_checkpointing_policy={nn.TransformerEncoderLayer},
                sharding_strategy="FULL_SHARD",
            )
        trainer = L.Trainer(
            logger=logger,
            callbacks=[
                every_epoch_checkpoint_callback,
                best_checkpoint_callback,
                lr_monitor,
            ],
            **training_args["trainer_args"],
        )
    else:
        trainer = L.Trainer(
            logger=logger,
            callbacks=[
                every_epoch_checkpoint_callback,
                best_checkpoint_callback,
                lr_monitor,
            ],
            **training_args["trainer_args"],
        )

    if trainer.global_rank == 0:
        train_folder_name = trainer.logger.log_dir
        os.makedirs(train_folder_name, exist_ok=True)
        OmegaConf.save(cfg, f"{train_folder_name}/config.yaml")

        # Extract and print version number
        version_match = os.path.basename(train_folder_name)
        print("\n" + "=" * 50)
        print(f"Lightning Logs Version: {version_match}")
        print(f"Log Directory: {train_folder_name}")
        print("=" * 50 + "\n")

    print(f"Reloading from {training_args['resume_training_path']}")
    trainer.fit(
        lightning_model,
        train_loader,
        val_loader,
        ckpt_path=training_args["resume_training_path"]
        if training_args["resume_training_path"] is not None
        else None,
    )
