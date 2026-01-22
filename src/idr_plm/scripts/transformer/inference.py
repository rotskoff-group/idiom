import torch
import hydra
import lightning as L
from omegaconf import OmegaConf
from mole.nn.transformer.module import LightningModel
from rdkit import Chem
import h5py
import pickle
import numpy as np
from mole.nn.transformer.module import (
    sample_components_from_bidirectional_transformer,
    sample_components_from_autoregressive_transformer,
    sample_components_from_transfusion_transformer,
    sample_structure_from_transfusion_transformer,
    sample_smiles_structure_from_mixed_seq_transformer,
)
from mole.nn.transformer.tokenizer import BasicSmilesTokenizer
from mole.utils.token import aggregate_tokens_hdf5
from mole.utils.sampler import TokenSampler


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="inference")
def main(cfg) -> None:
    model_args = cfg["model"]
    inference_args = cfg["inference"]
    training_args = cfg["training"]

    L.seed_everything(**training_args["seed_args"])

    device = "cuda:0"

    dataset_ptr = h5py.File(inference_args["dataset_filename"], "r")
    token_info = aggregate_tokens_hdf5(dataset_ptr)

    lightning_model = LightningModel(
        model_args=model_args, token_info=token_info, training_args=training_args
    )
    lightning_model.load_model_from_checkpoint(inference_args["checkpoint_path"])
    lightning_model.to(device)

    # Instantiate the sampler
    token_sampler = TokenSampler(**inference_args["sampler_args"])

    # Helps with knowing exact inference settings
    OmegaConf.save(cfg, f"{inference_args['savedir']}/inference_config.yaml")

    if inference_args["inference_mode"] == "bidirectional":
        if (
            inference_args["unmasking_mode"] == "sample"
            or inference_args["unmasking_mode"] == "all"
        ):
            ensemble_seq_lens = [i for i in range(1, 100 + 1)]
            batch_size = inference_args["batch_size"]
            total_size = batch_size * inference_args["num_batches"]

            for _, seq_len in enumerate(ensemble_seq_lens):
                print(
                    f"Using sequence length: {seq_len}, adding two tokens for start and stop"
                )
                seq_len = seq_len + 2
                structural_tokens = (
                    torch.ones((total_size, seq_len))
                    * token_info["input"]["TOK"]["TOK_PAD"]
                )
                structural_tokens = structural_tokens.long()
                masked_smiles_tokens = torch.ones((total_size, seq_len))
                masked_smiles_tokens[:, 0] = token_info["input"]["TOK"]["TOK_START"]
                masked_smiles_tokens[:, -1] = token_info["input"]["TOK"]["TOK_STOP"]
                masked_smiles_tokens[:, 1:-1] = token_info["input"]["TOK"]["TOK_MASK"]
                masked_smiles_tokens = masked_smiles_tokens.long()
                sequence_id = torch.ones((total_size, seq_len))
                sequence_id = sequence_id.long()

                structural_tokens = structural_tokens.to(device)
                masked_smiles_tokens = masked_smiles_tokens.to(device)
                sequence_id = sequence_id.to(device)

                with torch.no_grad():
                    unmasked_tokens = sample_components_from_bidirectional_transformer(
                        lightning_model,
                        structural_tokens,
                        masked_smiles_tokens,
                        sequence_id,
                        token_sampler=token_sampler,
                        inference_batch_size=batch_size,
                        unmasking_mode=inference_args["unmasking_mode"],
                    )
                torch.save(
                    unmasked_tokens, f"{inference_args['savedir']}/tst_{seq_len}.pt"
                )

        elif inference_args["unmasking_mode"] == "masked_left_to_right":
            batch_size = inference_args["batch_size"]
            total_size = batch_size * inference_args["num_batches"]
            start_tokens = (
                torch.ones((total_size, 1)) * token_info["input"]["TOK"]["TOK_START"]
            )
            start_tokens = start_tokens.long()
            # Not used
            structural_tokens = (
                torch.ones((total_size, 1)) * token_info["input"]["TOK"]["TOK_PAD"]
            )
            structural_tokens = structural_tokens.long()
            sequence_id = torch.ones((total_size, 1))
            sequence_id = sequence_id.long()

            # Send everything to the device
            start_tokens = start_tokens.to(device)
            structural_tokens = structural_tokens.to(device)
            sequence_id = sequence_id.to(device)
            with torch.no_grad():
                unmasked_tokens = sample_components_from_bidirectional_transformer(
                    lightning_model,
                    structural_tokens,
                    start_tokens,
                    sequence_id,
                    token_sampler=token_sampler,
                    inference_batch_size=batch_size,
                    unmasking_mode=inference_args["unmasking_mode"],
                )
                # output here is a tuple of lists, the first one is the sampled token ids by batch and the second is the
                #   token probabilities by batch
                torch.save(
                    unmasked_tokens, f"{inference_args['savedir']}/tst_{seq_len}.pt"
                )

    elif inference_args["inference_mode"] == "autoregressive":
        # import pdb; pdb.set_trace()
        # Define tokens at start
        _start_token = token_info["input"]["TOK"]["TOK_START"]
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
        batch_size = inference_args["batch_size"]
        total_size = batch_size * inference_args["num_batches"]

        # FH: Logic here accounts for the case where the user wants to continue
        # sampling from initialized SMILES tokens, not just unprompted sampling
        # from a start token. The input is expected to be tokens, not strings.
        if not inference_args["addn_args"]["use_input_smiles"]:
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
            smiles_file = inference_args["addn_args"]["smiles_path"]
            smi_tokens = np.load(smiles_file, allow_pickle=True)
            smi_tokens = [torch.tensor(s).long() for s in smi_tokens]
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

        with torch.no_grad():
            output = sample_components_from_autoregressive_transformer(
                transformer_model=lightning_model,
                structural_tokens=structural_tokens,
                smiles_tokens=smi_tokens,
                sequence_id=sequence_id,
                token_sampler=token_sampler,
                inference_batch_size=batch_size,
                use_input_smiles=inference_args["addn_args"]["use_input_smiles"],
            )
        # output here is a tuple of lists, the first one is the sampled token ids by batch and the second is the
        #   token probabilities by batch
        with open(f"{inference_args['savedir']}/tst_autoregressive.pkl", "wb") as f:
            pickle.dump(output, f)

    elif inference_args["inference_mode"] == "transfusion":
        batch_size = inference_args["batch_size"]
        total_size = batch_size * inference_args["num_batches"]
        run_diffusion = inference_args["addn_args"]["run_diffusion"]
        start_tokens = (
            torch.ones((total_size, 1)) * token_info["input"]["TOK"]["TOK_START"]
        )
        start_tokens = start_tokens.long()

        alphabet = np.load(inference_args["addn_args"]["alphabet"], allow_pickle=True)

        sequence_id = torch.ones((total_size, 1))
        sequence_id = sequence_id.long()

        # Send everything to the device
        start_tokens = start_tokens.to(device)
        sequence_id = sequence_id.to(device)
        with torch.no_grad():
            output = sample_components_from_transfusion_transformer(
                lightning_model,
                start_tokens,
                sequence_id,
                token_sampler=token_sampler,
                diff_config=inference_args["addn_args"],
                alphabet=alphabet,
                inference_batch_size=batch_size,
                run_diffusion=run_diffusion,
            )
        with open(f"{inference_args['savedir']}/tst_transfusion.pkl", "wb") as f:
            pickle.dump(output, f)

    elif inference_args["inference_mode"] == "transfusion_structure_only":
        batch_size = inference_args["batch_size"]
        # SMILES strings
        smiles_file = inference_args["addn_args"]["smiles_file"]
        alphabet = np.load(inference_args["addn_args"]["alphabet"], allow_pickle=True)
        alphabet_lookup = {v: k for k, v in enumerate(alphabet)}
        with open(smiles_file, "r") as f:
            smiles = f.readlines()
        smiles = [s.strip() for s in smiles]
        smiles = [Chem.CanonSmiles(s) for s in smiles]
        # Translate into tokens
        tokenizer = BasicSmilesTokenizer()
        tokens = [[alphabet_lookup[i] for i in tokenizer.tokenize(s)] for s in smiles]
        start_token = token_info["input"]["TOK"]["TOK_START"]
        struct_start = token_info["input"]["STRUCT"]["STRUCT_START"]
        tokens = [[start_token] + t + [struct_start] for t in tokens]
        tokens = [torch.tensor(t).long().to(device) for t in tokens]
        with torch.no_grad():
            output = sample_structure_from_transfusion_transformer(
                lightning_model,
                tokens,
                diff_config=inference_args["addn_args"],
                alphabet=alphabet,
                inference_batch_size=batch_size,
            )
        with open(
            f"{inference_args['savedir']}/tst_transfusion_structure.pkl", "wb"
        ) as f:
            pickle.dump(output, f)

    elif inference_args["inference_mode"] == "mixed_sequence":
        batch_size = inference_args["batch_size"]

        if not inference_args["addn_args"]["use_finished_smiles"]:
            total_size = batch_size * inference_args["num_batches"]
            start_tokens = (
                torch.ones((total_size, 1)) * token_info["input"]["TOK"]["TOK_START"]
            )
            smi_tokens = start_tokens.long()
            sequence_id = torch.ones((total_size, 1))
            sequence_id = sequence_id.long()
        elif (
            inference_args["addn_args"]["use_finished_smiles"]
            and inference_args["addn_args"]["smiles_path"] is not None
        ):
            # import pdb; pdb.set_trace()
            # Here, we are expecting a list of arrays which contain SMILES tokens
            smiles_file = inference_args["addn_args"]["smiles_path"]
            # A list of arrays, each containing the tokens for a smiles string
            smi_tokens = np.load(smiles_file, allow_pickle=True)
            smi_tokens = [torch.tensor(s).long() for s in smi_tokens]
            sequence_id = torch.ones(len(smi_tokens)).long()

        if ("avh_args" in inference_args["addn_args"]) and (
            inference_args["addn_args"]["avh_args"] is not None
        ):
            # We expect the avh_args dict to have everthing needed for avh-enhanced sampling of hte
            #   transformer
            avh_file = inference_args["addn_args"]["avh_args"]
            avh_args = np.load(avh_file, allow_pickle=True)
        else:
            avh_args = None

        alphabet = dataset_ptr["alphabet"][()]
        # Reformat into a numpy array of strings to allow for fast indexing
        alphabet = np.array([i.decode("utf-8") for i in alphabet])

        with torch.no_grad():
            output = sample_smiles_structure_from_mixed_seq_transformer(
                lightning_model,
                smi_tokens,
                sequence_id,
                token_sampler=token_sampler,
                inference_batch_size=batch_size,
                alphabet=alphabet,
                mode=inference_args["addn_args"]["mode"],
                struct_limit_method=inference_args["addn_args"]["struct_limit_method"],
                struct_limit_multiplier=inference_args["addn_args"][
                    "struct_limit_multiplier"
                ],
                use_finished_smiles=inference_args["addn_args"]["use_finished_smiles"],
                avh_args=avh_args,
            )
        # output here is a tuple of lists, the first one is the sampled token ids by batch and the second is the
        #   token probabilities by batch
        with open(
            f"{inference_args['savedir']}/tst_{inference_args['addn_args']['mode']}_mixed_seq.pkl",
            "wb",
        ) as f:
            pickle.dump(output, f)
