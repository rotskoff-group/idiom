import torch
import hydra
import lightning as L
from omegaconf import OmegaConf
import h5py
import pickle
import numpy as np
import os
import torch.multiprocessing as mp
from functools import reduce

from idiom.nn.transformer.module import LightningModel
from idiom.nn.transformer.utils.sampling import (
    sample_components_from_autoregressive_transformer,
)
from idiom.utils.token import aggregate_tokens_hdf5
from idiom.utils.sampler import TokenSampler
from idiom.utils.misc import (
    rearrange_sequence,
    extract_idr_with_indices,
)


def run_inference_on_gpu(
    gpu_id,
    num_batches_for_gpu,
    batch_size,
    model_args,
    training_args,
    inference_args,
    token_info,
    start_idx,
    end_idx,
    savedir,
    num_gpus,
    total_size,
    use_input_residues,
    residues_path,
    round_num=0,
):
    """Run inference on a specific GPU and save results to temporary file."""
    try:
        device = f"cuda:{gpu_id}"

        # Set seed for reproducibility. Offset by both gpu_id and round_num so
        # that each GPU and each retry round produces distinct sequences.
        seed_args = training_args["seed_args"].copy()
        seed_args["seed"] = seed_args.get("seed", 0) + gpu_id + round_num * num_gpus
        L.seed_everything(**seed_args)

        # Load model on this GPU
        lightning_model = LightningModel(
            model_args=model_args, token_info=token_info, training_args=training_args
        )
        lightning_model.load_model_from_checkpoint(inference_args["checkpoint_path"])
        lightning_model.to(device)

        # Instantiate the sampler
        token_sampler = TokenSampler(**inference_args["sampler_args"])

        # Load and slice the data for this GPU
        _start_token = token_info["input"]["TOK"]["TOK_START"]
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]

        if not use_input_residues:
            # Unprompted generation
            num_seqs = end_idx - start_idx
            start_tokens = torch.ones((num_seqs, 1)) * _start_token
            res_tokens_gpu = start_tokens.long()
            structural_tokens_gpu = (
                torch.ones((num_seqs, 1), dtype=torch.long) * _pad_token
            )
            sequence_id_gpu = torch.ones((num_seqs, 1), dtype=torch.long)
        else:
            # Prompted generation - load from file and slice
            res_tokens_full = np.load(residues_path, allow_pickle=True)
            res_tokens_full = [torch.tensor(s).long() for s in res_tokens_full]

            # Prepend start token if needed
            start_token_long = torch.tensor(_start_token, dtype=torch.long).unsqueeze(0)
            res_tokens_full = [
                torch.cat([start_token_long, seq], dim=0)
                if seq[0] != _start_token
                else seq
                for seq in res_tokens_full
            ]

            # Slice prompts for this GPU
            res_tokens_gpu = res_tokens_full[start_idx:end_idx]
            num_seqs = len(res_tokens_gpu)
            structural_tokens_gpu = (
                torch.ones(num_seqs, 1, dtype=torch.long) * _pad_token
            )
            sequence_id_gpu = torch.ones(num_seqs, dtype=torch.long)

        num_seqs = end_idx - start_idx
        print(
            f"GPU {gpu_id}: Processing sequences {start_idx}-{end_idx - 1} ({num_seqs} sequences, {num_batches_for_gpu} batches)"
        )

        with torch.no_grad():
            output = sample_components_from_autoregressive_transformer(
                transformer_model=lightning_model,
                structural_tokens=structural_tokens_gpu,
                res_tokens=res_tokens_gpu,
                sequence_id=sequence_id_gpu,
                token_sampler=token_sampler,
                inference_batch_size=batch_size,
                use_input_residues=inference_args["addn_args"]["use_input_residues"],
            )

        # Save results to temporary file
        temp_file = f"{savedir}/gpu_{gpu_id}_temp.pkl"
        with open(temp_file, "wb") as f:
            pickle.dump((gpu_id, output), f)

        print(f"GPU {gpu_id}: Completed inference, saved to {temp_file}")

    except Exception as e:
        print(f"GPU {gpu_id}: Error occurred - {str(e)}")
        import traceback

        traceback.print_exc()
        # Save None to indicate failure
        temp_file = f"{savedir}/gpu_{gpu_id}_temp.pkl"
        with open(temp_file, "wb") as f:
            pickle.dump((gpu_id, None), f)


def run_inference_on_gpu_length_filtered(
    gpu_id,
    batch_size,
    model_args,
    training_args,
    inference_args,
    token_info,
    num_seqs_needed,
    savedir,
    num_gpus,
    use_input_residues,
    residues_path,
    seq_length,
    seq_length_range,
    alphabet,
):
    """Load model once on a GPU and generate batches until num_seqs_needed sequences
    with IDR length within seq_length ± seq_length_range are collected.

    Saves a flat list of (valid_tokens, valid_probs) to the temp file, rather than
    the batched format used by run_inference_on_gpu.
    """
    try:
        device = f"cuda:{gpu_id}"
        base_seed = training_args["seed_args"].get("seed", 0)

        # Initial seed
        seed_args = training_args["seed_args"].copy()
        seed_args["seed"] = base_seed + gpu_id
        L.seed_everything(**seed_args)

        # Load model ONCE
        lightning_model = LightningModel(
            model_args=model_args, token_info=token_info, training_args=training_args
        )
        lightning_model.load_model_from_checkpoint(inference_args["checkpoint_path"])
        lightning_model.to(device)

        token_sampler = TokenSampler(**inference_args["sampler_args"])

        _start_token = token_info["input"]["TOK"]["TOK_START"]
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]

        # Load prompts once if using prompted generation
        if use_input_residues:
            res_tokens_all = np.load(residues_path, allow_pickle=True)
            res_tokens_all = [torch.tensor(s).long() for s in res_tokens_all]
            start_token_long = torch.tensor(_start_token, dtype=torch.long).unsqueeze(0)
            res_tokens_all = [
                torch.cat([start_token_long, seq], dim=0)
                if seq[0] != _start_token
                else seq
                for seq in res_tokens_all
            ]
            num_prompts = len(res_tokens_all)

        valid_tokens = []
        valid_probs = []
        batch_num = 0

        while len(valid_tokens) < num_seqs_needed:
            batch_num += 1

            # Re-seed each batch so successive batches produce different sequences.
            # Stride by num_gpus so GPU seeds never collide across the same batch_num.
            L.seed_everything(base_seed + gpu_id + batch_num * num_gpus)

            if not use_input_residues:
                res_batch = torch.ones((batch_size, 1), dtype=torch.long) * _start_token
                structural_batch = (
                    torch.ones((batch_size, 1), dtype=torch.long) * _pad_token
                )
                seq_id_batch = torch.ones((batch_size, 1), dtype=torch.long)
            else:
                # Cycle through the prompt file so every prompt gets reused evenly
                start = ((batch_num - 1) * batch_size) % num_prompts
                indices = [(start + i) % num_prompts for i in range(batch_size)]
                res_batch = [res_tokens_all[i] for i in indices]
                structural_batch = (
                    torch.ones(batch_size, 1, dtype=torch.long) * _pad_token
                )
                seq_id_batch = torch.ones(batch_size, dtype=torch.long)

            with torch.no_grad():
                batch_token_batches, batch_prob_batches = (
                    sample_components_from_autoregressive_transformer(
                        transformer_model=lightning_model,
                        structural_tokens=structural_batch,
                        res_tokens=res_batch,
                        sequence_id=seq_id_batch,
                        token_sampler=token_sampler,
                        inference_batch_size=batch_size,
                        use_input_residues=use_input_residues,
                    )
                )

            flat_tokens = list(reduce(lambda x, y: x + y, batch_token_batches))
            flat_probs = list(reduce(lambda x, y: x + y, batch_prob_batches))

            n_before = len(valid_tokens)
            for tok, prob in zip(flat_tokens, flat_probs):
                idr_len = _idr_length_from_tokens(tok, alphabet)
                if (
                    idr_len is not None
                    and abs(idr_len - seq_length) <= seq_length_range
                ):
                    valid_tokens.append(tok)
                    valid_probs.append(prob)
                    if len(valid_tokens) >= num_seqs_needed:
                        break

            n_accepted = len(valid_tokens) - n_before
            print(
                f"GPU {gpu_id} batch {batch_num}: accepted {n_accepted}/{len(flat_tokens)} "
                f"(need {max(0, num_seqs_needed - len(valid_tokens))} more)"
            )

        valid_tokens = valid_tokens[:num_seqs_needed]
        valid_probs = valid_probs[:num_seqs_needed]

        temp_file = f"{savedir}/gpu_{gpu_id}_temp.pkl"
        with open(temp_file, "wb") as f:
            pickle.dump((gpu_id, (valid_tokens, valid_probs)), f)

        print(f"GPU {gpu_id}: Done, collected {len(valid_tokens)} valid sequences")

    except Exception as e:
        print(f"GPU {gpu_id}: Error occurred - {str(e)}")
        import traceback

        traceback.print_exc()
        temp_file = f"{savedir}/gpu_{gpu_id}_temp.pkl"
        with open(temp_file, "wb") as f:
            pickle.dump((gpu_id, None), f)


def _dispatch_and_collect(
    num_gpus,
    total_size,
    batch_size,
    model_args,
    training_args,
    inference_args,
    token_info,
    savedir,
    use_input_residues,
    residues_path,
    round_num=0,
):
    """Dispatch one round of inference across GPUs and return batched token/prob lists."""
    sequences_per_gpu = total_size // num_gpus
    remainder = total_size % num_gpus

    if num_gpus > 1:
        print(f"Distributing {total_size} sequences across {num_gpus} GPUs:")
    else:
        print(f"Processing {total_size} sequences on 1 GPU:")

    processes = []
    for gpu_id in range(num_gpus):
        if gpu_id < remainder:
            start_idx = gpu_id * (sequences_per_gpu + 1)
            end_idx = start_idx + sequences_per_gpu + 1
        else:
            start_idx = (
                remainder * (sequences_per_gpu + 1)
                + (gpu_id - remainder) * sequences_per_gpu
            )
            end_idx = start_idx + sequences_per_gpu

        print(f"GPU {gpu_id}, start idx {start_idx}, end idx {end_idx}")

        num_seqs_for_gpu = end_idx - start_idx
        batches_for_gpu = (num_seqs_for_gpu + batch_size - 1) // batch_size
        print(
            f"  GPU {gpu_id}: {num_seqs_for_gpu} sequences ({batches_for_gpu} batches)"
        )

        p = mp.Process(
            target=run_inference_on_gpu,
            args=(
                gpu_id,
                batches_for_gpu,
                batch_size,
                model_args,
                training_args,
                inference_args,
                token_info,
                start_idx,
                end_idx,
                savedir,
                num_gpus,
                total_size,
                use_input_residues,
                residues_path,
                round_num,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All GPU processes completed, collecting results...")

    results = []
    for gpu_id in range(num_gpus):
        temp_file = f"{savedir}/gpu_{gpu_id}_temp.pkl"
        try:
            with open(temp_file, "rb") as f:
                results.append(pickle.load(f))
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not load results from GPU {gpu_id}: {e}")
            results.append((gpu_id, None))

    results.sort(key=lambda x: x[0])

    all_tokens = []
    all_probs = []
    for gpu_id, output_tuple in results:
        if output_tuple is not None:
            tokens, probs = output_tuple
            all_tokens.extend(tokens)
            all_probs.extend(probs)
        else:
            print(f"Warning: GPU {gpu_id} returned no results")

    return all_tokens, all_probs


def _dispatch_length_filtered(
    num_gpus,
    total_size,
    batch_size,
    model_args,
    training_args,
    inference_args,
    token_info,
    savedir,
    use_input_residues,
    residues_path,
    seq_length,
    seq_length_range,
    alphabet,
):
    """Spawn one persistent process per GPU; each loads the model once and loops
    internally until it has collected its share of valid-length sequences."""
    sequences_per_gpu = total_size // num_gpus
    remainder = total_size % num_gpus

    if num_gpus > 1:
        print(f"Distributing {total_size} target sequences across {num_gpus} GPUs:")
    else:
        print(f"Collecting {total_size} length-filtered sequences on 1 GPU:")

    processes = []
    for gpu_id in range(num_gpus):
        # First 'remainder' GPUs collect one extra sequence
        num_seqs_needed = sequences_per_gpu + (1 if gpu_id < remainder else 0)
        print(f"  GPU {gpu_id}: needs {num_seqs_needed} valid sequences")

        p = mp.Process(
            target=run_inference_on_gpu_length_filtered,
            args=(
                gpu_id,
                batch_size,
                model_args,
                training_args,
                inference_args,
                token_info,
                num_seqs_needed,
                savedir,
                num_gpus,
                use_input_residues,
                residues_path,
                seq_length,
                seq_length_range,
                alphabet,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All GPU processes completed, collecting results...")

    results = []
    for gpu_id in range(num_gpus):
        temp_file = f"{savedir}/gpu_{gpu_id}_temp.pkl"
        try:
            with open(temp_file, "rb") as f:
                results.append(pickle.load(f))
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not load results from GPU {gpu_id}: {e}")
            results.append((gpu_id, None))

    results.sort(key=lambda x: x[0])

    all_tokens = []
    all_probs = []
    for gpu_id, output_tuple in results:
        if output_tuple is not None:
            tokens, probs = output_tuple  # flat lists from length-filtered worker
            all_tokens.extend(tokens)
            all_probs.extend(probs)
        else:
            print(f"Warning: GPU {gpu_id} returned no results")

    return all_tokens, all_probs


def _idr_length_from_tokens(token_seq, alphabet):
    """Return the IDR region length for a token sequence, or None if no IDR found."""
    if token_seq is None:
        return None
    seq_chars = "".join(alphabet[t] for t in token_seq if t <= 22)
    idr_seq, _, _ = extract_idr_with_indices(seq_chars)
    return len(idr_seq) if idr_seq else None


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="inference")
def main(cfg) -> None:
    model_args = cfg["model"]
    inference_args = cfg["inference"]
    training_args = cfg["training"]

    L.seed_everything(**training_args["seed_args"])

    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = inference_args.get("use_multi_gpu", False) and num_gpus > 1

    if use_multi_gpu:
        print(f"Multi-GPU inference enabled: Using {num_gpus} GPUs")
    else:
        print("Single GPU inference: Using 1 GPU")
        num_gpus = 1

    dataset_ptr = h5py.File(inference_args["dataset_filename"], "r")
    token_info = aggregate_tokens_hdf5(dataset_ptr)

    # Helps with knowing exact inference settings
    OmegaConf.save(cfg, f"{inference_args['savedir']}/inference_config.yaml")

    if inference_args["inference_mode"] == "autoregressive":
        batch_size = inference_args["batch_size"]

        if not inference_args["addn_args"]["use_input_residues"]:
            # Unprompted generation - total size from num_batches
            total_batches = inference_args["num_batches"]
            total_size = batch_size * total_batches
        elif (
            inference_args["addn_args"]["use_input_residues"]
            and inference_args["addn_args"]["residues_path"] is not None
        ):
            # Prompted generation - total size from prompt file
            residues_file = inference_args["addn_args"]["residues_path"]
            res_tokens_check = np.load(residues_file, allow_pickle=True)
            total_size = len(res_tokens_check)
            print(f"Loaded {total_size} input sequences from {residues_file}")

        # Set multiprocessing start method to 'spawn' (NOT 'fork')
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        savedir = inference_args["savedir"]
        use_input_residues = inference_args["addn_args"]["use_input_residues"]
        residues_path = inference_args["addn_args"]["residues_path"]

        # Optional length filtering: ++inference.addn_args.seq_length and
        # ++inference.addn_args.seq_length_range
        seq_length = inference_args["addn_args"].get("seq_length", None)
        seq_length_range = inference_args["addn_args"].get("seq_length_range", None)

        if seq_length is not None and seq_length_range is not None:
            # Length-filtered generation: each GPU process loads the model once and
            # loops internally until it has collected its quota of valid sequences.
            print(
                f"\nLength-filtered generation enabled: "
                f"target IDR length {seq_length} ± {seq_length_range} residues"
            )

            # Load alphabet in the main process and pass it to workers
            shard = h5py.File(inference_args["dataset_filename"], "r")
            alphabet = [x.decode("utf-8") for x in shard["alphabet"][()]]
            shard.close()

            all_valid_tokens, all_valid_probs = _dispatch_length_filtered(
                num_gpus,
                total_size,
                batch_size,
                model_args,
                training_args,
                inference_args,
                token_info,
                savedir,
                use_input_residues,
                residues_path,
                seq_length,
                seq_length_range,
                alphabet,
            )

            print(
                f"\nLength-filtered generation complete: {len(all_valid_tokens)} sequences "
                f"with IDR length {seq_length} ± {seq_length_range}"
            )
            # Wrap in a list so reduce(lambda x,y: x+y, output[0]) flattens correctly
            output = ([all_valid_tokens], [all_valid_probs])

        else:
            # Standard single-round inference
            round_token_batches, round_prob_batches = _dispatch_and_collect(
                num_gpus,
                total_size,
                batch_size,
                model_args,
                training_args,
                inference_args,
                token_info,
                savedir,
                use_input_residues,
                residues_path,
            )
            output = (round_token_batches, round_prob_batches)

        # Save raw output
        with open(f"{savedir}/tst_autoregressive.pkl", "wb") as f:
            pickle.dump(output, f)

        print(f"Inference complete. Results saved to {savedir}/tst_autoregressive.pkl")

        # Export sequences to FASTA files
        print("\nExporting sequences to FASTA files...")

        # Load alphabet from shard file
        precomputed_shard = h5py.File(inference_args["dataset_filename"], "r")
        alphabet = precomputed_shard["alphabet"][()]
        alphabet = [x.decode("utf-8") for x in alphabet]

        # Load the pickle file the same way as idp_to_fasta.py
        pkl_path = f"{savedir}/tst_autoregressive.pkl"
        with open(pkl_path, "rb") as f:
            inference_out = pickle.load(f)

        # Flatten the token lists (same approach as idp_to_fasta.py)
        seqs_tokens = list(reduce(lambda x, y: x + y, inference_out[0]))

        # Convert tokens to characters using alphabet
        sequences = []
        for seq in seqs_tokens:
            if seq is None:
                sequences.append(None)
            else:
                # Convert tokens to characters, filtering out padding tokens (>22)
                seq_chars = "".join(alphabet[token] for token in seq if token <= 22)
                sequences.append(seq_chars)

        # Filter out None sequences
        sequences = [s for s in sequences if s is not None]

        # Extract IDRs and full sequences
        idr_sequences = []
        full_sequences_with_indices = []

        for i, seq in enumerate(sequences):
            # Extract IDR
            idr_seq, idr_start, idr_end = extract_idr_with_indices(seq)

            if idr_seq:  # Only keep sequences with IDR
                idr_sequences.append((i, idr_seq))
                # Rearrange full sequence
                full_seq = rearrange_sequence(seq)
                full_sequences_with_indices.append((i, full_seq, idr_start, idr_end))

        print(
            # f"Found {len(idr_sequences)} sequences with disordered regions out of {len(sequences)}"
        )

        # Save FASTA files
        # File 1: IDR regions only
        idr_fasta = os.path.join(savedir, "generated_idrs.fasta")
        with open(idr_fasta, "w") as f:
            for seq_idx, idr_seq in idr_sequences:
                f.write(f">seq_{seq_idx:06d}\n")
                f.write(f"{idr_seq}\n")

        # File 2: Full sequences with IDR indices in header
        full_fasta = os.path.join(savedir, "generated_full.fasta")
        with open(full_fasta, "w") as f:
            for seq_idx, full_seq, idr_start, idr_end in full_sequences_with_indices:
                f.write(f">seq_{seq_idx:06d}_IDR_{idr_start + 1}-{idr_end}\n")
                f.write(f"{full_seq}\n")
        print(
            # f"Successfully wrote {len(full_sequences_with_indices)} full sequences to {full_fasta}"
        )

        precomputed_shard.close()


if __name__ == "__main__":
    main()
