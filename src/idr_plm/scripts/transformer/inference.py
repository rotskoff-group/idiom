import torch
import hydra
import lightning as L
from omegaconf import OmegaConf
import h5py
import pickle
import numpy as np
import os
import torch.multiprocessing as mp

from idr_plm.nn.transformer.module import LightningModel
from idr_plm.nn.transformer.module import sample_components_from_autoregressive_transformer
from idr_plm.utils.token import aggregate_tokens_hdf5
from idr_plm.utils.sampler import TokenSampler


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
    use_input_smiles,
    smiles_path,
):
    """Run inference on a specific GPU and save results to temporary file."""
    try:
        device = f"cuda:{gpu_id}"

        # Set seed for reproducibility
        seed_args = training_args["seed_args"].copy()
        # Use gpu_id offset so each GPU doesn't generate the same sequences
        seed_args["seed"] = seed_args.get("seed", 0) + gpu_id
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

        if not use_input_smiles:
            # Unprompted generation
            num_seqs = end_idx - start_idx
            start_tokens = torch.ones((num_seqs, 1)) * _start_token
            smi_tokens_gpu = start_tokens.long()
            structural_tokens_gpu = (
                torch.ones((num_seqs, 1), dtype=torch.long) * _pad_token
            )
            sequence_id_gpu = torch.ones((num_seqs, 1), dtype=torch.long)
        else:
            # Prompted generation - load from file and slice
            smi_tokens_full = np.load(smiles_path, allow_pickle=True)
            smi_tokens_full = [torch.tensor(s).long() for s in smi_tokens_full]

            # Prepend start token if needed
            start_token_long = torch.tensor(_start_token, dtype=torch.long).unsqueeze(0)
            smi_tokens_full = [
                torch.cat([start_token_long, seq], dim=0)
                if seq[0] != _start_token
                else seq
                for seq in smi_tokens_full
            ]

            # Slice prompts for this GPU
            smi_tokens_gpu = smi_tokens_full[start_idx:end_idx]
            num_seqs = len(smi_tokens_gpu)
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
                smiles_tokens=smi_tokens_gpu,
                sequence_id=sequence_id_gpu,
                token_sampler=token_sampler,
                inference_batch_size=batch_size,
                use_input_smiles=inference_args["addn_args"]["use_input_smiles"],
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
        # import pdb; pdb.set_trace()
        # Define tokens at start
        _start_token = token_info["input"]["TOK"]["TOK_START"]
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
        batch_size = inference_args["batch_size"]

        # Prepare input sequences
        # FH: Logic here accounts for the case where the user wants to continue
        # sampling from initialized SMILES tokens, not just unprompted sampling
        # from a start token. The input is expected to be tokens, not strings.
        if not inference_args["addn_args"]["use_input_smiles"]:
            # Unprompted generation - use num_batches parameter
            total_batches = inference_args["num_batches"]
            total_size = batch_size * total_batches

            start_tokens = torch.ones((total_size, 1)) * _start_token
            start_tokens = start_tokens.long()
            # Not used
            structural_tokens = torch.ones((total_size, 1)) * _pad_token
            structural_tokens = structural_tokens.long()
            sequence_id = torch.ones((total_size, 1))
            sequence_id = sequence_id.long()

            # FH: Don't send to device until the batch is created within the sampling method
            smi_tokens = start_tokens
            # smi_tokens = smi_tokens.to(device)
            # structural_tokens = structural_tokens.to(device)
            # sequence_id = sequence_id.to(device)

        elif (
            inference_args["addn_args"]["use_input_smiles"]
            and inference_args["addn_args"]["smiles_path"] is not None
        ):
            # Prompted generation - use actual number of input sequences
            smiles_file = inference_args["addn_args"]["smiles_path"]
            smi_tokens = np.load(smiles_file, allow_pickle=True)
            smi_tokens = [torch.tensor(s).long() for s in smi_tokens]

            total_size = len(smi_tokens)
            print(f"Loaded {total_size} input sequences from {smiles_file}")

            sequence_id = torch.ones(len(smi_tokens)).long()
            structural_tokens = torch.ones(len(smi_tokens), 1) * _pad_token

            # jxliu2: Prepend start token to each sequence in smi_tokens after loading from smiles_file
            start_token_long = torch.tensor(_start_token, dtype=torch.long).unsqueeze(
                0
            )  # Needs to be 1D for torch.cat()
            # smi_tokens = [torch.cat([start_token_long, seq], dim=0) for seq in smi_tokens]
            smi_tokens = [
                torch.cat([start_token_long, seq], dim=0)
                if seq[0] != _start_token
                else seq
                for seq in smi_tokens
            ]
            # import ipdb; ipdb.set_trace()

            # type cast
            structural_tokens = structural_tokens.long()
            sequence_id = sequence_id.long()

        # Set multiprocessing start method to 'spawn' (NOT 'fork')
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # If spawn is already set, continue
            pass

        # Use multiprocessing for both single and multi-GPU inference
        processes = []

        # Distribute sequences among GPUs
        sequences_per_gpu = total_size // num_gpus  # floor div
        remainder = total_size % num_gpus  # modulo

        if num_gpus > 1:
            print(f"Distributing {total_size} sequences across {num_gpus} GPUs:")
        else:
            print(f"Processing {total_size} sequences on 1 GPU:")

        for gpu_id in range(num_gpus):
            # First 'remainder' GPUs get one extra sequence
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
            # Calculate batches needed for this GPU
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
                    inference_args["savedir"],
                    num_gpus,
                    total_size,
                    inference_args["addn_args"]["use_input_smiles"],
                    inference_args["addn_args"]["smiles_path"],
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        print("All GPU processes completed, collecting results...")

        # Collect results from temporary files
        results = []
        for gpu_id in range(num_gpus):
            temp_file = f"{inference_args['savedir']}/gpu_{gpu_id}_temp.pkl"
            try:
                with open(temp_file, "rb") as f:
                    results.append(pickle.load(f))
                # Clean up temporary file
                os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not load results from GPU {gpu_id}: {e}")
                results.append((gpu_id, None))

        # Sort by GPU ID and combine results
        results.sort(key=lambda x: x[0])

        # Combine outputs
        all_tokens = []
        all_probs = []
        for gpu_id, output_tuple in results:
            if output_tuple is not None:
                tokens, probs = output_tuple
                all_tokens.extend(tokens)
                all_probs.extend(probs)
            else:
                print(f"Warning: GPU {gpu_id} returned no results")

        output = (all_tokens, all_probs)

        # output here is a tuple of lists, the first one is the sampled token ids by batch and the second is the
        #   token probabilities by batch
        with open(f"{inference_args['savedir']}/tst_autoregressive.pkl", "wb") as f:
            pickle.dump(output, f)

        print(
            f"Inference complete. Results saved to {inference_args['savedir']}/tst_autoregressive.pkl"
        )


if __name__ == "__main__":
    main()
