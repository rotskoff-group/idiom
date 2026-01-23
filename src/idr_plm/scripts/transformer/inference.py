import torch
import hydra
import lightning as L
from omegaconf import OmegaConf
from idr_plm.nn.transformer.module import LightningModel
# from rdkit import Chem
import h5py
import pickle
import numpy as np
from idr_plm.nn.transformer.module import (
    # sample_components_from_bidirectional_transformer,
    sample_components_from_autoregressive_transformer,
    # sample_components_from_transfusion_transformer,
    # sample_structure_from_transfusion_transformer,
    # sample_smiles_structure_from_mixed_seq_transformer,
)
from idr_plm.nn.transformer.tokenizer import BasicSmilesTokenizer
from idr_plm.utils.token import aggregate_tokens_hdf5
from idr_plm.utils.sampler import TokenSampler


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

    if inference_args["inference_mode"] == "autoregressive":
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
