import numpy as np
import torch_geometric


def get_mol_conf_indices(
    dataset: torch_geometric.data.Dataset, split_indices: list, max_conformers: int = 1
):
    """
    Given a dataset and a list of molecule indices, return the molecule and conformer indices.

    Parameters:
        dataset (torch_geometric.data.Dataset): The dataset containing molecular data.
        split_indices (list): List of molecule indices to process.
        max_conformers (int, optional): Maximum number of conformers per molecule. Defaults to 1.

    Returns:
        tuple: A tuple containing two lists:
            - mol_indices: List of molecule indices repeated for each conformer.
            - conf_indices: List of conformer indices for each molecule.
    """
    mol_indices = []
    conf_indices = []
    for i_mol in split_indices:
        n_confs = min(dataset.n_conformers[i_mol], max_conformers)
        mol_indices.extend([int(i_mol)] * n_confs)
        conf_indices.extend(range(n_confs))
    return mol_indices, conf_indices


def generate_mol_conf_indices(
    dataset: torch_geometric.data.Dataset,
    sets_to_run: list,
    splits_path: str,
    max_conformers: int = 1,
    batch_size: int = 1,
):
    """
    Generate molecule and conformer indices for the specified dataset splits.

    Parameters:
        dataset (torch_geometric.data.Dataset): The dataset containing molecular data.
        sets_to_run (list): List of dataset splits to process (e.g., ["train", "val", "test"]).
        splits_path (str): Path to the file containing split indices.
        max_conformers (int, optional): Maximum number of conformers per molecule. Defaults to 1.
        batch_size (int, optional): Number of molecules per batch. Defaults to 1.

    Returns:
        tuple: A tuple containing train, validation, and test indices as lists of of pairs
        molecule-conformer indices.
    """
    splits = np.load(splits_path, allow_pickle=True).item()
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]
    if "train" in sets_to_run:
        mol_indices, conf_indices = get_mol_conf_indices(
            dataset, train_split, max_conformers=max_conformers
        )
        train_indices = [
            [mol_indices[i : i + batch_size], conf_indices[i : i + batch_size]]
            for i in range(0, len(mol_indices), batch_size)
        ]
    else:
        train_indices = None
    if "val" in sets_to_run:
        mol_indices, conf_indices = get_mol_conf_indices(
            dataset, val_split, max_conformers=max_conformers
        )
        val_indices = [
            [mol_indices[i : i + batch_size], conf_indices[i : i + batch_size]]
            for i in range(0, len(mol_indices), batch_size)
        ]
    else:
        val_indices = None
    if "test" in sets_to_run:
        mol_indices, conf_indices = get_mol_conf_indices(
            dataset, test_split, max_conformers=max_conformers
        )
        test_indices = [
            [mol_indices[i : i + batch_size], conf_indices[i : i + batch_size]]
            for i in range(0, len(mol_indices), batch_size)
        ]
    else:
        test_indices = None

    return train_indices, val_indices, test_indices
