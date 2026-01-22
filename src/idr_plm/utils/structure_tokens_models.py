import pickle
import numpy as np
import h5py
from ase import Atoms


soap_species = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Bi"]


def load_avh_combinations(subset_path: str, mode: str) -> np.ndarray:
    """
    Load the AVH combinations from a h5 clustering subset file.
    """
    with h5py.File(subset_path, "r") as f:
        if mode == "all":
            avh_combinations = f["all_avh_combinations"][:]
        elif mode == "clusterable":
            avh_combinations = f["clusterable_avh_combinations"][:]
        else:
            raise ValueError("Mode must be 'all' or 'clusterable'.")
    return avh_combinations


def get_clustering_models_and_indices(
    models_path: str,
    clusterable_combinations: np.ndarray,
    all_avh_combinations: np.ndarray,
    dataset: str,
) -> tuple:
    counter = len(all_avh_combinations)
    print(f"Total combinations: {counter}")
    print(f"Clusterable combinations: {len(clusterable_combinations)}")

    combinations_keys = [
        f"{combination[0]}_{combination[1]}_{combination[2]}"
        for combination in clusterable_combinations
    ]

    # Load the structure token models
    print("Loading clustering models...")
    clustering_models = []
    counter = 0
    global_codebook_number = []  # denotes the starting index of for a specific combination and is used to conver from "local" cluster number (withon and avh) to global cluster number

    for i_combination, combination_key in enumerate(combinations_keys):
        try:
            with open(
                f"{models_path}/{combination_key}_kmeans.pkl", "rb"
            ) as f:  # models/geom_clustering_models
                kmeans = pickle.load(f)
                if str(type(kmeans)) == "<class 'sklearn.cluster._kmeans.KMeans'>":
                    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)
                    clustering_models.append(kmeans)
                    global_codebook_number.append(counter)
                    counter += len(kmeans.cluster_centers_)
                else:
                    clustering_models.append(None)
                    global_codebook_number.append(counter)
                    counter += 1
        except Exception as e:
            print(f"Warning: Did not load model for {combination_key}: {e}")
            print("Assigning one cluster to it.")
            clustering_models.append(None)
            global_codebook_number.append(counter)
            counter += 1
    for combination in all_avh_combinations[len(clusterable_combinations) :]:
        clustering_models.append(None)
        global_codebook_number.append(counter)
        counter += 1

    print("Clustering models loaded.")

    return clustering_models, global_codebook_number


def rdkit_to_ase(rdmol):
    conf = rdmol.GetConformer()
    positions = []
    symbols = []

    for atom in rdmol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        symbols.append(atom.GetSymbol())

    return Atoms(symbols=symbols, positions=positions)
