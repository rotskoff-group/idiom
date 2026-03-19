import hydra
from functools import reduce
import h5py
import numpy as np
from typing import Callable
from multiprocessing import Pool
from tqdm import tqdm

from idr_plm.nn.transformer.utils import tokenizer as tokenmodule
from idr_plm.nn.transformer.utils.tokenizer import CharTokenizer
from idr_plm.nn.transformer.generators import input_generators as input_generators
from idr_plm.nn.transformer.generators import target_generators as target_generators


def determine_alphabet(residues: list[str], tokenizer: CharTokenizer) -> list[str]:
    """Determines all unique tokens represented in a set of strings

    Args:
        residues: list[str]
            List of residue sequences to determine the alphabet over
        tokenizer: CharTokenizer
            Tokenizer object used to find unique tokens from the strings

    Returns:
        alphabet: list[str]
            List of all unique tokens as determined from the given residue sequences
    """
    token_sets = [set(tokenizer.tokenize(res)) for res in tqdm(residues)]
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

    residues = h5py.File(precompute_args["residues_file"], "r")["residues"]
    residues = [res.decode("utf-8") for res in residues]

    # Get the tokenizer
    try:
        tokenizer = getattr(tokenmodule, precompute_args["tokenizer"])()
    except Exception:
        raise ValueError(
            f"Tokenizer {precompute_args['tokenizer']} not implemented/recognized!"
        )

    alp = precompute_args["alphabet"]
    if alp is None:
        print("Determining alphabet based on residues")
        alphabet = determine_alphabet(residues, tokenizer)
    else:
        print(f"Loading the following: {alp}")
        alphabet = np.load(alp, allow_pickle=True)
        alphabet = [str(x) for x in alphabet]

    input_generator = getattr(input_generators, precompute_args["input_generator"])(
        residues, tokenizer, alphabet, **precompute_args["input_generator_addn_args"]
    )

    target_generator = getattr(target_generators, precompute_args["target_generator"])(
        residues,
        tokenizer,
        alphabet,
        targets,
        **precompute_args["target_generator_addn_args"],
    )

    # Process the data in parallel
    num_processes = precompute_args["num_processes"]
    print("Starting parallel runs...")

    if precompute_args["precompute_data_format"] == "residues_only":
        processed_inputs = run_process_parallel(
            input_generator.transform, {}, num_processes, residues
        )

        processed_targets = run_process_parallel(
            target_generator.transform, {}, num_processes, residues, targets
        )
        # FH: Dataset is precomputed and intended only to train the model on residue data tasks,
        #   such as generating residue sequences or predicting properties from residues

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

        # Compute the sequence id here over the tokenized residues (inputs only)
        sequence_id = np.array(processed_inputs)
        input_pad_token = input_metadata["ctrl_tokens"]["TOK_PAD"]
        # True for nonpadding, false for padding
        sequence_id[sequence_id != input_pad_token] = 1
        sequence_id[sequence_id == input_pad_token] = 0

        struct_tokens = (
            np.ones(processed_inputs.shape) * input_metadata["ctrl_tokens"]["TOK_PAD"]
        )

        with h5py.File(precompute_args["output_file"], "w") as f:
            f.create_dataset("res_tokens", data=processed_inputs)
            f.create_dataset("targets", data=processed_targets)
            f.create_dataset("residues", data=residues)
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

    else:
        raise ValueError(
            f"Unrecognized precompute format provided, {precompute_args['precompute_data_format']}"
        )
