import h5py
from torch.utils.data import Dataset
from typing import Optional
import numpy as np
import torch
import os


def get_hdf5_fn(dataset_filename):
    """Gets the dataset from a coarse-grained model
    Args:
        dataset_filename (str): The name of the file where the dataset is saved
    Returns:
        dset_hdf5: A dictionary with the dataset
    """

    def get_hdf5_data():
        if os.path.isdir(dataset_filename):
            all_h5_files = [
                os.path.join(dataset_filename, f)
                for f in os.listdir(dataset_filename)
                if f.endswith(".h5")
            ]
            all_h5_files = sorted(all_h5_files)
            return [h5py.File(f, "r") for f in all_h5_files]
        else:
            return h5py.File(dataset_filename, "r")

    return get_hdf5_data


# def create_dataset_from_path(dataset_filename, model_config):
#     """Creates a dataset from a path to a hdf5 file
#     Args:
#         dataset_filename (str): The name of the file where the dataset is saved
#         nn_config (dict): A dictionary with the configuration for the neural network
#     Returns:
#         dataset: A (subclassed) PyTorch Dataset
#     """
#     dataset = getattr(clm.models, model_config["dataset"])
#     get_hdf5_data = get_hdf5_fn(dataset_filename)

#     dataset = dataset(get_hdf5_data=get_hdf5_data,
#                       data_in_memory=model_config["data_in_memory"],
#                       config=model_config)
#     return dataset


def split_data_subsets(
    dataset: Dataset,
    splits: Optional[str],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple:
    """Splits the dataset using indices from passed file
    Args:
        dataset: The dataset to split
        splits: The path to the numpy file with the indices for the splits
        train_size: The fraction of the dataset to use for training
        val_size: The fraction of the dataset to use for validation
        test_size: The fraction of the dataset to use for testing
    """
    if splits is not None:
        print(f"Splitting data using indices from {splits}")
        split_indices = np.load(splits, allow_pickle=True).item()
        train, val, test = (
            split_indices["train"],
            split_indices["val"],
            split_indices["test"],
        )
        return (
            torch.utils.data.Subset(dataset, train),
            torch.utils.data.Subset(dataset, val),
            torch.utils.data.Subset(dataset, test),
        )
    else:
        assert train_size + val_size + test_size == 1
        print(
            f"Splitting data using {train_size} train, {val_size} val, {test_size} test"
        )
        train, val, test = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        return train, val, test
