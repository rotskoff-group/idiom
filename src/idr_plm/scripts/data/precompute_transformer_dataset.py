import hydra
import os
from functools import reduce
from mole.nn.transformer import tokenizer as tokenmodule
from mole.nn.transformer.tokenizer import BasicSmilesTokenizer, CharTokenizer
from mole.nn.transformer import input_generators as input_generators
from mole.nn.transformer import target_generators as target_generators
import h5py
import numpy as np
from typing import Callable
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import Chem
from pathlib import Path
import torch
import torch.nn.functional as F
from mole.utils.conflow_featurization import geom_atom_mapping as atom_mapping


def read_from_conflow_dataset(
    conflow_dataset_path: Path,
    read_structure_tokens: bool = False,
    avh_mode: bool = False,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    """Reads the SMILES strings from the ConFlow dataset

    Args:
        conflow_dataset_path: Path
            Path to the ConFlow dataset

    Returns:
        smiles: list[str]
            List of SMILES strings
        transformer_splits: list[str]
            List of splits for the transformer model
    """
    smiles = []
    transformer_splits = {}
    conformer_counter = 0
    this_split_min = 0
    if read_structure_tokens:
        structure_tokens = []
    else:
        structure_tokens = None
    if avh_mode:
        avhs = []
    else:
        avhs = None

    max_st_token = 0
    max_st_len = 0
    max_avh = torch.tensor([0, 0, 0])  # [atom_type, valency, hybridization]

    conflow_dataset_path = Path(conflow_dataset_path)
    for split in ["train", "val"]:
        print(f"Reading {split} split...")
        split_dir = conflow_dataset_path / split
        if not split_dir.exists():
            continue
        for file in tqdm(split_dir.glob("*.pt")):
            data = torch.load(file)
            n_conformers = data["pos"].shape[0]
            conformer_counter += n_conformers
            if avh_mode:
                remapped_avh = data[
                    "node_attr"
                ][
                    :, [-2, -1, 5]
                ].int()  # [atom_type, valency, hybridization] see src/mole/utils/structure_tokens_models.py
                remapped_avh = remapped_avh[
                    remapped_avh[:, 0] != 1
                ]  # select only heavy atoms
                remapped_avh[:, 0] = remapped_avh[:, 0].apply_(
                    lambda x: atom_mapping[x]
                )  # avhs remapped to contiguous mapping
                max_avh = torch.maximum(max_avh, torch.max(remapped_avh, dim=0)[0])
            for _ in range(n_conformers):
                smiles.append(data["smiles"].encode("utf-8"))
                if avh_mode:
                    avhs.append(remapped_avh)
            if read_structure_tokens:
                for st in data["structure_tokens"]:
                    st = st.int()
                    structure_tokens.append(st)
                    if max(st) > max_st_token:
                        max_st_token = max(st)
                if len(data["structure_tokens"][0]) > max_st_len:
                    max_st_len = len(st)

        transformer_splits[split] = np.arange(this_split_min, conformer_counter)
        this_split_min = conformer_counter
    transformer_splits["test"] = np.array(
        []
    )  # change if test set is processed as train and val
    struct_pad = int(max_st_token + 1)
    max_tokens = {
        "struct_pad": struct_pad,
        "struct_stop": struct_pad + 1,
        "struct_mask": struct_pad + 2,
        "struct_start": struct_pad + 3,
        "struct_max_length": max_st_len,
        "atom_padding_idx": int(max_avh[0]) + 1,
        "valency_padding_idx": int(max_avh[1]) + 1,
        "hybrid_padding_idx": int(max_avh[2]) + 1,
    }
    return smiles, transformer_splits, structure_tokens, avhs, max_tokens


def determine_alphabet(
    smiles: list[str], tokenizer: BasicSmilesTokenizer | CharTokenizer
) -> list[str]:
    """Determines all unique tokens represented in a set of strings

    Args:
        smiles: list[str]
            List of SMILES strings to determine the alphabet over
        tokenizer: BasicSmilesTokenizer
            Tokenizer object used to find unique tokens from the strings

    Returns:
        alphabet: list[str]
            List of all unique tokens as determined from the given SMILES strings
    """
    token_sets = [set(tokenizer.tokenize(smi)) for smi in tqdm(smiles)]
    final_set = list(reduce(lambda x, y: x.union(y), token_sets))
    alphabet = sorted(final_set)
    return alphabet


def run_process_parallel(
    f: Callable,
    f_addn_args: dict[str],
    num_processes: int,
    *data_args: tuple[np.ndarray],
) -> list:
    """Runs function f in parallel with num_processes processes

    Args:
        f: Callable
            Function to be run in parallel
        f_addn_args: dict[str]
            Additional keyword arguments required by f
        num_processes: int
            Number of processes to run in parallel
        data_args: tuple[np.ndarray]
            Arguments to be passed to f, should be a tuple of arrays.

    Returns:
        result: list
            List of results from running f in parallel over each chunk in data_args

    Notes:
        For data_args, if there is more than one array, then the arrays should have the
        same first dimension size and index correspondence between elements.
        For multiple arrays, it is assumed they are passed in the order that the
        elements would be passed to function f, i.e.

        ([x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]) -> f(x1, y1, z1), f(x2, y2, z2), ...
    """
    assert len(data_args) > 0
    pool = Pool(processes=num_processes)
    if len(data_args) > 1:
        data_input = zip(*data_args)
        result = pool.starmap_async(f, data_input)
    else:
        result = pool.map_async(f, data_args[0])

    pool.close()
    pool.join()
    return result.get()


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="precompute")
def main(cfg) -> None:
    precompute_args = cfg["precompute"]

    if precompute_args["target_file"] is not None:
        targets_h5 = h5py.File(precompute_args["target_file"], "r")
        targets = targets_h5["targets"]
    else:
        targets = None
    if precompute_args["precompute_data_format"] == "SMILES_and_VQVAE_struct":
        conflow_dset_path = precompute_args["conflow_dset_path"]
        read_structure_tokens = "struct" in precompute_args["precompute_data_format"]
        avh_mode = "AVH" in precompute_args["input_generator"]
        smiles, transformer_splits, structure_tokens, avhs, max_tokens = (
            read_from_conflow_dataset(
                conflow_dset_path,
                read_structure_tokens=read_structure_tokens,
                avh_mode=avh_mode,
            )
        )
        # Create tranformer data directory and save splits there
        os.makedirs(conflow_dset_path + "/transformer", exist_ok=True)
        np.save(
            conflow_dset_path + "transformer/transformer_splits.npy", transformer_splits
        )
    else:
        smiles = h5py.File(precompute_args["smiles_file"], "r")["smiles"]

    if precompute_args["canonicalize_smiles"]:
        print("Canonicalizing SMILES strings...")
        # Canonicalize the SMILES strings. This is important, especially for the
        #   order of the atoms in the molecule that is then used to determin the
        #   dihedral angles by traversing the molecule. See the compute_dihedrals() method
        #   in mole.nn.transformer.input_generators.py
        smiles = [Chem.CanonSmiles(smi.decode("utf-8")) for smi in smiles]
    else:
        print("NOT canonicalizing SMILES strings...")
        smiles = [smi.decode("utf-8") for smi in smiles]

    # Get the tokenizer
    try:
        tokenizer = getattr(tokenmodule, precompute_args["tokenizer"])()
    except Exception:
        raise ValueError(
            f"Tokenizer {precompute_args['tokenizer']} not implemented/recognized!"
        )

    alp = precompute_args["alphabet"]
    if alp is None:
        print("Determining alphabet based on SMILES")
        alphabet = determine_alphabet(smiles, tokenizer)
    else:
        print(f"Loading the following: {alp}")
        alphabet = np.load(alp, allow_pickle=True)
        alphabet = [str(x) for x in alphabet]

    if (
        "max_tokens" in locals()
    ):  # if max_tokens is defined, we know we are doing avh and we can set the struct_token_info
        precompute_args["input_generator_addn_args"] = {"struct_token_info": max_tokens}

    input_generator = getattr(input_generators, precompute_args["input_generator"])(
        smiles, tokenizer, alphabet, **precompute_args["input_generator_addn_args"]
    )

    target_generator = getattr(target_generators, precompute_args["target_generator"])(
        smiles,
        tokenizer,
        alphabet,
        targets,
        **precompute_args["target_generator_addn_args"],
    )
    # Process the data in parallel
    num_processes = precompute_args["num_processes"]
    print("Starting parallel runs...")

    if precompute_args["precompute_data_format"] == "SMILES_only":
        processed_inputs = run_process_parallel(
            input_generator.transform, {}, num_processes, smiles
        )

        processed_targets = run_process_parallel(
            target_generator.transform, {}, num_processes, smiles, targets
        )
        # FH: Dataset is precomputed and intended only to train the model on SMILES data tasks,
        #   such as generating SMILES or predicting properties from SMILES

        # Convert all targets and inputs into numpy arrays for encoding
        processed_inputs = np.array(list(map(lambda x: np.array(x), processed_inputs)))
        processed_targets = np.array(
            list(map(lambda x: np.array(x), processed_targets))
        )

        input_metadata = {
            "source_size": input_generator.get_size(),
            "ctrl_tokens": input_generator.get_ctrl_tokens(),
            "max_seq_len": input_generator.get_max_seq_len(),
        }
        target_metadata = {
            "target_size": target_generator.get_size(),
            "ctrl_tokens": target_generator.get_ctrl_tokens(),
            "max_seq_len": target_generator.get_max_seq_len(),
        }

        print("target_metadata", target_metadata)
        print("input_metadata", input_metadata)

        # Compute the sequence id here over the tokenized SMILES (inputs only)
        sequence_id = np.array(processed_inputs)
        input_pad_token = input_metadata["ctrl_tokens"]["TOK_PAD"]
        # True for nonpadding, false for padding
        sequence_id[sequence_id != input_pad_token] = 1
        sequence_id[sequence_id == input_pad_token] = 0

        struct_tokens = (
            np.ones(processed_inputs.shape) * input_metadata["ctrl_tokens"]["TOK_PAD"]
        )

        with h5py.File(precompute_args["output_file"], "w") as f:
            f.create_dataset("smi_tokens", data=processed_inputs)
            f.create_dataset("targets", data=processed_targets)
            f.create_dataset("smiles", data=smiles)
            f.create_dataset("alphabet", data=alphabet)
            f.create_dataset("sequence_id", data=sequence_id)
            f.create_dataset("structural_tokens", data=struct_tokens)

            inp_meta = f.create_group("input_metadata")
            inp_meta.create_dataset("source_size", data=input_metadata["source_size"])
            inp_meta.create_dataset("max_seq_len", data=input_metadata["max_seq_len"])
            inp_meta_ctrl_tokens = inp_meta.create_group("ctrl_tokens")
            for k, v in input_metadata["ctrl_tokens"].items():
                inp_meta_ctrl_tokens.create_dataset(k, data=v)

            tar_meta = f.create_group("target_metadata")
            tar_meta.create_dataset("target_size", data=target_metadata["target_size"])
            tar_meta.create_dataset("max_seq_len", data=target_metadata["max_seq_len"])
            tar_meta_ctrl_tokens = tar_meta.create_group("ctrl_tokens")
            for k, v in target_metadata["ctrl_tokens"].items():
                tar_meta_ctrl_tokens.create_dataset(k, data=v)

    elif precompute_args["precompute_data_format"] == "SMILES_and_struct":
        # FH: Dataset is precomputed to include both SMILES and structural data for unified approaches
        #   like transfusion. For now, assume featurizing molecules using the dihedral angles computed
        #   from the canonicalized SMILES string
        # Need to pad the structural data correctly to ensure everything is the correct length
        # The smiles input is always processed
        processed_inputs = run_process_parallel(
            input_generator.transform, {}, num_processes, smiles
        )
        processed_tokens_input = np.array([x[0] for x in processed_inputs])
        processed_tokens_target = np.array([x[1] for x in processed_inputs])
        processed_struct = np.array([x[2] for x in processed_inputs])
        processed_struct_indices = np.array([x[3] for x in processed_inputs])
        assert (
            len(processed_tokens_input)
            == len(processed_tokens_target)
            == len(processed_struct)
            == len(smiles)
            == len(processed_struct_indices)
        )

        # Need to do a filtering stage here to remove cases where the dihedral calculation failed
        #   and the structural data is empty
        invalid_embeddings = processed_struct == -100
        invalid_embeddings_mask = (
            np.sum(invalid_embeddings, axis=1) == processed_struct.shape[-1]
        )
        processed_tokens_input = processed_tokens_input[~invalid_embeddings_mask]
        processed_tokens_target = processed_tokens_target[~invalid_embeddings_mask]
        processed_struct = processed_struct[~invalid_embeddings_mask]
        processed_struct_indices = processed_struct_indices[~invalid_embeddings_mask]
        smiles = [
            smiles[i] for i in range(len(smiles)) if not invalid_embeddings_mask[i]
        ]

        input_metadata = {
            "source_size": input_generator.get_size(),
            "ctrl_tokens": input_generator.get_ctrl_tokens(),
            "max_seq_len": input_generator.get_max_seq_len(),
        }

        # Construct the sequence id based on the tokenized input. This mapping is based on the mha layer implementation,
        #   where 1 is used for non-padding SMILES tokens, 2 is used for structural tokens, and 0 is used for padding
        struct_token = input_metadata["ctrl_tokens"]["STRUCT"]
        pad_token = input_metadata["ctrl_tokens"]["TOK_PAD"]

        # They should have the same masking for padding and structure tokens, so can do seq_id construction based
        #   on the input tokens only
        assert np.all(
            (processed_tokens_input == pad_token)
            == (processed_tokens_target == pad_token)
        )
        assert np.all(
            (processed_tokens_input == struct_token)
            == (processed_tokens_target == struct_token)
        )

        # Structure start and structure end embedded using standard embedding layer
        pad_token_mask_input = processed_tokens_input == pad_token
        structure_mask = processed_tokens_input == struct_token
        sequence_id = np.zeros(processed_tokens_input.shape)
        sequence_id[pad_token_mask_input] = 0
        sequence_id[structure_mask] = 2
        sequence_id[~(pad_token_mask_input + structure_mask)] = 1

        with h5py.File(precompute_args["output_file"], "w") as f:
            f.create_dataset(
                "token_input", data=processed_tokens_input
            )  # For next-token prediction
            f.create_dataset(
                "token_target", data=processed_tokens_target
            )  # For next-token prediction
            f.create_dataset(
                "structure", data=processed_struct
            )  # For diffusion over structure
            f.create_dataset(
                "angle_indices", data=processed_struct_indices
            )  # Indices for mapping angles
            f.create_dataset("smiles", data=smiles)  # Smiles strings
            f.create_dataset("alphabet", data=alphabet)  # Alphabet of tokens
            f.create_dataset(
                "sequence_id", data=sequence_id
            )  # Sequence id for transformer

            inp_meta = f.create_group("input_metadata")
            inp_meta.create_dataset("source_size", data=input_metadata["source_size"])
            inp_meta.create_dataset("max_seq_len", data=input_metadata["max_seq_len"])
            inp_meta_ctrl_tokens = inp_meta.create_group("ctrl_tokens")
            for k, v in input_metadata["ctrl_tokens"].items():
                inp_meta_ctrl_tokens.create_dataset(k, data=v)

    elif precompute_args["precompute_data_format"] == "SMILES_and_struct_2":
        # Alternative implementation that adds the STRUCT_START token back into the sequence and uses only the input
        # The smiles input is always processed
        processed_inputs = run_process_parallel(
            input_generator.transform, {}, num_processes, smiles
        )
        processed_tokens = np.array([x[0] for x in processed_inputs])
        processed_struct = np.array([x[1] for x in processed_inputs])
        processed_struct_indices = np.array([x[2] for x in processed_inputs])
        assert (
            len(processed_tokens)
            == len(processed_struct)
            == len(smiles)
            == len(processed_struct_indices)
        )

        # Need to do a filtering stage here to remove cases where the dihedral calculation failed
        #   and the structural data is empty
        invalid_embeddings = processed_struct == -100
        invalid_embeddings_mask = (
            np.sum(invalid_embeddings, axis=1) == processed_struct.shape[-1]
        )
        processed_tokens = processed_tokens[~invalid_embeddings_mask]
        processed_struct = processed_struct[~invalid_embeddings_mask]
        processed_struct_indices = processed_struct_indices[~invalid_embeddings_mask]
        smiles = [
            smiles[i] for i in range(len(smiles)) if not invalid_embeddings_mask[i]
        ]

        input_metadata = {
            "source_size": input_generator.get_size(),
            "ctrl_tokens": input_generator.get_ctrl_tokens(),
            "max_seq_len": input_generator.get_max_seq_len(),
        }

        struct_token = input_metadata["ctrl_tokens"]["STRUCT"]
        smi_pad_token = input_metadata["ctrl_tokens"]["TOK_PAD"]
        struct_start = input_metadata["ctrl_tokens"]["STRUCT_START"]
        struct_end = input_metadata["ctrl_tokens"]["STRUCT_END"]

        # Construct the sequence_id
        sequence_id = np.zeros(processed_tokens.shape)
        pad_token_mask = processed_tokens == smi_pad_token
        struct_mask = processed_tokens == struct_token
        struct_start_mask = processed_tokens == struct_start
        struct_end_mask = processed_tokens == struct_end
        sequence_id[pad_token_mask] = 0
        sequence_id[struct_mask] = 2
        sequence_id[struct_start_mask] = 2
        sequence_id[struct_end_mask] = 2
        sequence_id[
            ~(pad_token_mask + struct_mask + struct_start_mask + struct_end_mask)
        ] = 1

        with h5py.File(precompute_args["output_file"], "w") as f:
            f.create_dataset("token_input", data=processed_tokens)
            f.create_dataset("structure", data=processed_struct)
            f.create_dataset("angle_indices", data=processed_struct_indices)
            f.create_dataset("smiles", data=smiles)
            f.create_dataset("alphabet", data=alphabet)
            f.create_dataset("sequence_id", data=sequence_id)

            inp_meta = f.create_group("input_metadata")
            inp_meta.create_dataset("source_size", data=input_metadata["source_size"])
            inp_meta.create_dataset("max_seq_len", data=input_metadata["max_seq_len"])
            inp_meta_ctrl_tokens = inp_meta.create_group("ctrl_tokens")
            for k, v in input_metadata["ctrl_tokens"].items():
                inp_meta_ctrl_tokens.create_dataset(k, data=v)

    elif precompute_args["precompute_data_format"] == "SMILES_and_VQVAE_struct":
        # Precompute mode where the transformer model is trained bi-directionally with masking to predict smiles and structure tokens
        # The bi-directional nature of the model is encoded in the construction of the sequence_id which follows the packed_seq convention,
        # see chem-language-models mha.py implementation for more details
        #
        # This precompute assumes you already have the structure tokens computed for the given set of SMILES via a trained VQVAE model
        # assert "struct_token_file" in precompute_args
        # structure_tokens = h5py.File(precompute_args["struct_token_file"], "r")[
        #    "structure_tokens"
        # ]
        print("Padding and reformatting...")
        for i in range(len(structure_tokens)):
            aux = torch.stack(
                [structure_tokens[i], avhs[i][:, 0], avhs[i][:, 1], avhs[i][:, 2]]
            )
            structure_tokens[i] = F.pad(
                aux,
                (
                    0,
                    max_tokens["struct_pad"] - aux.shape[1],
                    0,
                    0,
                ),
                value=max_tokens["struct_pad"],
            )
        structure_tokens = torch.stack(structure_tokens)
        structure_tokens = structure_tokens.numpy()
        processed_inputs = run_process_parallel(
            input_generator.transform, {}, num_processes, smiles, structure_tokens
        )
        processed_tokens = np.array([x[0] for x in processed_inputs])
        processed_struct = np.array([x[1] for x in processed_inputs])
        input_metadata = {
            "source_size": input_generator.get_size(),
            "ctrl_tokens": input_generator.get_ctrl_tokens(),
            "max_seq_len": input_generator.get_max_seq_len(),
        }
        smi_pad_token = input_metadata["ctrl_tokens"]["TOK_PAD"]
        sequence_id = np.zeros(processed_tokens.shape)
        sequence_id = ~(processed_tokens == smi_pad_token)
        sequence_id = sequence_id.astype("int")

        print("Writing to file...")

        with h5py.File(precompute_args["output_file"], "w") as f:
            f.create_dataset("smi_tokens", data=processed_tokens)
            f.create_dataset("structure_tokens", data=processed_struct)
            f.create_dataset("smiles", data=smiles)
            f.create_dataset("alphabet", data=alphabet)
            f.create_dataset("sequence_id", data=sequence_id)

            inp_meta = f.create_group("input_metadata")
            inp_meta.create_dataset("source_size", data=input_metadata["source_size"])
            inp_meta.create_dataset("max_seq_len", data=input_metadata["max_seq_len"])
            inp_meta_ctrl_tokens = inp_meta.create_group("ctrl_tokens")
            for k, v in input_metadata["ctrl_tokens"].items():
                inp_meta_ctrl_tokens.create_dataset(k, data=v)

    elif precompute_args["precompute_data_format"] == "IDA_dataset":
        # GR TODO: for safety, store the structure tokens in an h5 indexed by smiles to ensure correct ordering
        assert "struct_token_file" in precompute_args
        structure_tokens = h5py.File(precompute_args["struct_token_file"], "r")[
            "structure_tokens"
        ]
        structure_tokens = np.array(structure_tokens)
        # read the coordinates from file
        assert "coords_file" in precompute_args
        coords = h5py.File(precompute_args["coords_file"], "r")["coords"]
        coords = np.array(coords)  # (N_molecule, N_atoms, 3)

        processed_inputs = run_process_parallel(
            input_generator.transform,
            {},
            num_processes,
            smiles,
            structure_tokens,
            coords,
        )
        processed_tokens = np.array([x[0] for x in processed_inputs])
        processed_struct = np.array([x[1] for x in processed_inputs])
        processed_coords = np.array([x[2] for x in processed_inputs])

        input_metadata = {
            "source_size": input_generator.get_size(),
            "ctrl_tokens": input_generator.get_ctrl_tokens(),
            "max_seq_len": input_generator.get_max_seq_len(),
        }
        smi_pad_token = input_metadata["ctrl_tokens"]["TOK_PAD"]
        sequence_id = np.zeros(processed_tokens.shape)
        sequence_id = ~(processed_tokens == smi_pad_token)
        sequence_id = sequence_id.astype("int")

        with h5py.File(precompute_args["output_file"], "w") as f:
            f.create_dataset("smi_tokens", data=processed_tokens)
            f.create_dataset("structure_tokens", data=processed_struct)
            f.create_dataset("coords", data=processed_coords)
            f.create_dataset("smiles", data=smiles)
            f.create_dataset("alphabet", data=alphabet)
            f.create_dataset("sequence_id", data=sequence_id)

            inp_meta = f.create_group("input_metadata")
            inp_meta.create_dataset("source_size", data=input_metadata["source_size"])
            inp_meta.create_dataset("max_seq_len", data=input_metadata["max_seq_len"])
            inp_meta_ctrl_tokens = inp_meta.create_group("ctrl_tokens")
            for k, v in input_metadata["ctrl_tokens"].items():
                inp_meta_ctrl_tokens.create_dataset(k, data=v)

    else:
        raise ValueError(
            f"Unrecognized precompute format provided, {precompute_args['precompute_data_format']}"
        )
