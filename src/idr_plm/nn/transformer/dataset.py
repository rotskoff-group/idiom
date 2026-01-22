import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable
from torch.nn.utils.rnn import pad_sequence


# Fairly sure that stacking is the default collate fn of the dataloader,
#   but will specify it explicitly just in case
def geometric_transformer_bidirec_collate_fn(
    data: tuple[torch.Tensor],
) -> tuple[torch.Tensor]:
    (structural_tokens, smi_tokens, masked_smi_tokens, sequence_id, loss_mask) = zip(
        *data
    )

    structural_tokens = torch.stack(structural_tokens)
    smi_tokens = torch.stack(smi_tokens)
    masked_smi_tokens = torch.stack(masked_smi_tokens)
    sequence_id = torch.stack(sequence_id)
    loss_mask = torch.stack(loss_mask)

    return (structural_tokens, smi_tokens, masked_smi_tokens, sequence_id, loss_mask)


def geometric_transformer_mixed_seq_collate_fn(
    data: tuple[torch.Tensor],
) -> tuple[torch.Tensor]:
    zipped_data = tuple(zip(*data))
    mode = zipped_data[0][0]
    if mode == "smiles_only":
        _, smi_tokens, masked_smi_tokens, sequence_id, loss_mask = zipped_data
        smi_tokens = torch.stack(smi_tokens)
        masked_smi_tokens = torch.stack(masked_smi_tokens)
        sequence_id = torch.stack(sequence_id)
        loss_mask = torch.stack(loss_mask)
        return (mode, smi_tokens, masked_smi_tokens, sequence_id, loss_mask)
    elif mode in ["smiles_and_struct", "smiles_struct_avh"]:
        (
            _,
            smi_tokens,
            masked_smi_tokens,
            struct_tokens,
            masked_struct_tokens,
            sequence_id,
            loss_mask_smi,
            loss_mask_struct,
        ) = zipped_data
        smi_tokens = torch.stack(smi_tokens)
        masked_smi_tokens = torch.stack(masked_smi_tokens)
        struct_tokens = torch.stack(struct_tokens)
        masked_struct_tokens = torch.stack(masked_struct_tokens)
        sequence_id = torch.stack(sequence_id)
        loss_mask_smi = torch.stack(loss_mask_smi)
        loss_mask_struct = torch.stack(loss_mask_struct)
        return (
            mode,
            smi_tokens,
            masked_smi_tokens,
            struct_tokens,
            masked_struct_tokens,
            sequence_id,
            loss_mask_smi,
            loss_mask_struct,
        )


def geometric_transformer_autoreg_collate_fn(
    data: tuple[torch.Tensor],
) -> tuple[torch.Tensor]:
    (
        structural_tokens,
        src_tokens,
        src_key_pad_mask,
        tgt_tokens,
        tgt_key_pad_mask,
        sequence_id,
    ) = zip(*data)

    structural_tokens = torch.stack(structural_tokens)
    src_tokens = torch.stack(src_tokens)
    src_key_pad_mask = torch.stack(src_key_pad_mask)
    tgt_tokens = torch.stack(tgt_tokens)
    tgt_key_pad_mask = torch.stack(tgt_key_pad_mask)
    sequence_id = torch.stack(sequence_id)

    return (
        structural_tokens,
        src_tokens,
        src_key_pad_mask,
        tgt_tokens,
        tgt_key_pad_mask,
        sequence_id,
    )


def transfusion_collate_fn(data: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    (token_input, token_target, struct_input, struct_mask, sequence_id) = zip(*data)

    token_input = torch.stack(token_input)
    token_target = torch.stack(token_target)
    struct_input = torch.stack(struct_input)
    sequence_id = torch.stack(sequence_id)
    struct_mask = torch.stack(struct_mask)

    return (token_input, token_target, struct_input, struct_mask, sequence_id)


def transformer_era_collate_fn(data):
    (tokens, masks, energies, ref_logps) = zip(*data)

    tokens = torch.cat(tokens, axis=0)
    masks = torch.cat(masks, axis=0)
    energies = torch.cat(energies, axis=0)
    ref_logps = torch.cat(ref_logps, axis=0)

    return (tokens, masks, energies, ref_logps)


def transformer_online_collate_fn(data):
    (tokens, masks) = zip(*data)

    tokens = torch.stack(tokens, dim=0)
    masks = torch.stack(masks, dim=0)

    return (tokens, masks)


class GeometricTransformerDatasetBidirec(Dataset):
    def __init__(self, get_hdf5_data: Callable, data_in_memory: bool = True) -> None:
        self.get_hdf5_data = get_hdf5_data
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5["smi_tokens"])
        self.mask_token_smi = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_MASK"][()]
        del t_hdf5

    def __len__(self) -> int:
        return self.length

    def _sample_from_beta_linear30(self) -> float:
        if np.random.rand() < 0.8:
            # Draw from Beta(3, 9) distribution
            mask_rate = np.random.beta(3, 9)
        else:
            # Draw from a linear distribution
            mask_rate = np.random.uniform(0, 1)

        return mask_rate

    def _get_mask_indices(self, sequence_length: int) -> np.ndarray:
        mask_rate = self._sample_from_beta_linear30()
        num_to_mask = int(mask_rate * sequence_length)
        # Ensure at least one token is masked (necessary for mask computation)
        num_to_mask = max(1, num_to_mask)
        mask_indices = np.random.choice(sequence_length, num_to_mask, replace=False)
        return mask_indices

    def open_hdf5(self) -> None:
        self.t_hdf5 = self.get_hdf5_data()

        self.structural_tokens = self.t_hdf5["structural_tokens"]
        self.smi_tokens = self.t_hdf5["smi_tokens"]
        self.sequence_id = self.t_hdf5["sequence_id"]

        if self.data_in_memory:
            self.structural_tokens = self.structural_tokens[:]
            self.smi_tokens = self.smi_tokens[:]
            self.sequence_id = self.sequence_id[:]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        if not hasattr(self, "t_hdf5"):
            self.open_hdf5()

        structural_tokens = torch.tensor(self.structural_tokens[idx], dtype=torch.long)
        smi_tokens = torch.tensor(self.smi_tokens[idx], dtype=torch.long)
        sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)

        non_special_indices = torch.where(
            smi_tokens < self.t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]
        )[0]
        seq_len = non_special_indices.shape[0]

        non_pad_mask_indices = torch.tensor(
            self._get_mask_indices(seq_len), dtype=torch.long
        )
        mask_indices = non_special_indices[non_pad_mask_indices]

        masked_smi_tokens = torch.clone(smi_tokens)
        masked_smi_tokens[mask_indices] = self.mask_token_smi

        # Additionally, always mask the stop token (start token is given)
        stop_token_indices = torch.where(
            smi_tokens == self.t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_STOP"][()]
        )[0]
        masked_smi_tokens[stop_token_indices] = self.mask_token_smi
        loss_mask = masked_smi_tokens == self.mask_token_smi

        return (
            structural_tokens,
            smi_tokens,
            masked_smi_tokens,
            sequence_id,
            loss_mask,
        )


# FH: Horrible name, but will only deprecate old version once this is tested
class GeometricTransformerMixedSequenceDataset(Dataset):
    def __init__(
        self,
        get_hdf5_data: Callable,
        data_in_memory: bool = True,
        data_mode: str = "smiles_and_struct",
        process_mode: str = "masked",
        fixed_masking_on_end_tokens: bool = True,
    ) -> None:
        """
        get_hdf5_data: Callable, a function that returns the hdf5 data pointer
        data_in_memory: bool, whether to load the data into memory
        data_mode: str, one of ['smiles_only', 'smiles_and_struct']
        process_mode: str, one of ['masked', 'autoreg'] where 'masked' is bidirectional modeling
        fixed_masking_on_end_tokens: bool, whether to always mask the end tokens. Default is True
        """
        assert data_mode in [
            "smiles_only",
            "smiles_and_struct",
            "smiles_struct_avh",
        ]
        self.get_hdf5_data = get_hdf5_data
        self.data_mode = data_mode
        self.process_mode = process_mode
        self.fixed_masking_on_end_tokens = fixed_masking_on_end_tokens
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5["smi_tokens"])
        # Unpack some useful metadata
        self.smiles_pad_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]
        self.smiles_start_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_START"][
            ()
        ]
        self.smiles_stop_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_STOP"][()]
        self.smiles_mask_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_MASK"][()]
        if self.data_mode in [
            "smiles_and_struct",
            "smiles_struct_avh",
        ]:
            self.structure_token = t_hdf5["input_metadata"]["ctrl_tokens"]["STRUCT"][()]
            self.structure_pad_token = t_hdf5["input_metadata"]["ctrl_tokens"][
                "STRUCT_PAD"
            ][()]
            self.structure_start_token = t_hdf5["input_metadata"]["ctrl_tokens"][
                "STRUCT_START"
            ][()]
            self.structure_stop_token = t_hdf5["input_metadata"]["ctrl_tokens"][
                "STRUCT_STOP"
            ][()]
            self.structure_mask_token = t_hdf5["input_metadata"]["ctrl_tokens"][
                "STRUCT_MASK"
            ][()]
        del t_hdf5

    def __len__(self) -> int:
        return self.length

    def _sample_from_beta_linear30(self) -> float:
        if np.random.rand() < 0.8:
            # Draw from Beta(3, 9) distribution
            mask_rate = np.random.beta(3, 9)
        else:
            # Draw from a linear distribution
            mask_rate = np.random.uniform(0, 1)

        return mask_rate

    def _get_mask_indices(self, sequence_length: int) -> np.ndarray:
        mask_rate = self._sample_from_beta_linear30()
        num_to_mask = int(mask_rate * sequence_length)
        # Ensure at least one token is masked (necessary for mask computation)
        num_to_mask = max(1, num_to_mask)
        mask_indices = np.random.choice(sequence_length, num_to_mask, replace=False)
        return mask_indices

    def open_hdf5(self) -> None:
        self.t_hdf5 = self.get_hdf5_data()
        if self.data_mode in [
            "smiles_and_struct",
            "smiles_struct_avh",
        ]:
            self.structure_tokens = self.t_hdf5["structure_tokens"]

        self.smi_tokens = self.t_hdf5["smi_tokens"]
        self.sequence_id = self.t_hdf5["sequence_id"]

        if self.data_in_memory:
            if self.data_mode in [
                "smiles_and_struct",
                "smiles_struct_avh",
            ]:
                self.structure_tokens = self.structure_tokens[:]

            self.smi_tokens = self.smi_tokens[:]
            self.sequence_id = self.sequence_id[:]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        if not hasattr(self, "t_hdf5"):
            self.open_hdf5()

        if (self.data_mode == "smiles_only") and (self.process_mode == "masked"):
            sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)
            smi_tokens = torch.tensor(self.smi_tokens[idx], dtype=torch.long)

            # Masking for the non-padding SMILES tokens
            non_special_indices = torch.where(smi_tokens < self.smiles_pad_token)[0]
            seq_len = non_special_indices.shape[0]

            non_pad_mask_indices = torch.tensor(
                self._get_mask_indices(seq_len), dtype=torch.long
            )
            mask_indices = non_special_indices[non_pad_mask_indices]

            masked_smi_tokens = torch.clone(smi_tokens)
            masked_smi_tokens[mask_indices] = self.smiles_mask_token

            if self.fixed_masking_on_end_tokens:
                # Additionally, always mask the stop token (start token is given)
                stop_token_indices = smi_tokens == self.smiles_stop_token
                masked_smi_tokens[stop_token_indices] = self.smiles_mask_token

            loss_mask = masked_smi_tokens == self.smiles_mask_token

            return (
                self.data_mode,
                smi_tokens,
                masked_smi_tokens,
                sequence_id,
                loss_mask,
            )

        elif (self.data_mode == "smiles_only") and (
            self.process_mode == "autoregressive"
        ):
            # FH: smiles_only is experimental right now, focusing on smiles_and_struct instead
            sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)
            smi_tokens = torch.tensor(self.smi_tokens[idx], dtype=torch.long)
            masked_smi_tokens = torch.clone(smi_tokens)
            loss_mask = sequence_id.clone()
            return (
                self.data_mode,
                smi_tokens,
                masked_smi_tokens,
                sequence_id,
                loss_mask,
            )

        elif (self.data_mode == "smiles_and_struct") and (
            self.process_mode == "masked"
        ):
            # Here, sequences are presumed to have the following structure:
            #   <TOK_START><SMILES><TOK_STOP><STRUCT><STRUCT_STOP>
            # Need to not only modify the masked smiles tokens to also mask out the <TOK_STOP> and <STRUCT_STOP> tokens,
            #   but also apply masking to the structural tokens
            sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)
            smi_tokens = torch.tensor(self.smi_tokens[idx], dtype=torch.long)
            struct_tokens = torch.tensor(self.structure_tokens[idx], dtype=torch.long)

            # Masking for the non-padding SMILES tokens
            non_special_smi_indices = torch.where(smi_tokens < self.smiles_pad_token)[0]
            smi_seq_len = non_special_smi_indices.shape[0]
            non_pad_smi_mask_indices = torch.tensor(
                self._get_mask_indices(smi_seq_len), dtype=torch.long
            )
            smi_mask_indices = non_special_smi_indices[non_pad_smi_mask_indices]
            masked_smi_tokens = torch.clone(smi_tokens)
            masked_smi_tokens[smi_mask_indices] = self.smiles_mask_token

            if self.fixed_masking_on_end_tokens:
                # Additionally, always mask the stop token (start token is given)
                stop_token_indices = smi_tokens == self.smiles_stop_token
                masked_smi_tokens[stop_token_indices] = self.smiles_mask_token

            loss_mask_smi = masked_smi_tokens == self.smiles_mask_token

            # Masking for the non-padding structural tokens
            non_special_struct_indices = torch.where(
                struct_tokens < self.structure_pad_token
            )[0]
            struct_seq_len = non_special_struct_indices.shape[0]
            non_pad_struct_mask_indices = torch.tensor(
                self._get_mask_indices(struct_seq_len), dtype=torch.long
            )
            struct_mask_indices = non_special_struct_indices[
                non_pad_struct_mask_indices
            ]
            masked_struct_tokens = torch.clone(struct_tokens)
            masked_struct_tokens[struct_mask_indices] = self.structure_mask_token

            if self.fixed_masking_on_end_tokens:
                # Additionally, always mask the stop token (start token is given)
                stop_token_indices = struct_tokens == self.structure_stop_token
                masked_struct_tokens[stop_token_indices] = self.structure_mask_token

            loss_mask_struct = masked_struct_tokens == self.structure_mask_token

            return (
                self.data_mode,
                smi_tokens,
                masked_smi_tokens,
                struct_tokens,
                masked_struct_tokens,
                sequence_id,
                loss_mask_smi,
                loss_mask_struct,
            )

        elif (self.data_mode == "smiles_and_struct") and (
            self.process_mode == "autoregressive"
        ):
            # FH: Because of the format of the data where the smiles token sequence has the following form:
            #   <TOK_START><SMILES><TOK_STOP><STRUCT><STRUCT_STOP>
            #   We need to do the standard shifting trick for autoregressive training
            #   The "masked" smi tokens go into the model and the regular smi_tokens are the
            #   target for the cross entropy loss
            # FH: TODO, is the sequence id correct here or are we off by one in length due to the token
            #   shifting in the autoregressive case? Probably doesn't have an effect on the model training
            #   because the loss mask takes care of this...
            sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)
            smi_tokens = torch.tensor(self.smi_tokens[idx], dtype=torch.long)
            struct_tokens = torch.tensor(self.structure_tokens[idx], dtype=torch.long)
            masked_smi_tokens = torch.clone(smi_tokens)
            masked_struct_tokens = torch.clone(struct_tokens)

            # Format the input sequence for cross entropy loss
            struct_token_indices = masked_smi_tokens == self.structure_token
            masked_smi_tokens[struct_token_indices.nonzero().max()] = (
                self.smiles_pad_token
            )
            # FH: More generally, set the last structure token to also be padding
            masked_struct_tokens[
                (masked_struct_tokens != self.structure_pad_token).nonzero().max()
            ] = self.structure_pad_token

            # Format the target sequence for cross entropy
            smi_tokens = torch.roll(smi_tokens, shifts=-1, dims=0)
            smi_tokens[-1] = self.smiles_pad_token

            # The loss mask here should select out only the smiles tokens, not the structural tokens
            loss_mask_smi = (smi_tokens != self.smiles_pad_token) & (
                smi_tokens != self.structure_token
            )
            loss_mask_struct = struct_tokens != self.structure_pad_token

            return (
                self.data_mode,
                smi_tokens,
                masked_smi_tokens,
                struct_tokens,
                masked_struct_tokens,
                sequence_id,
                loss_mask_smi,
                loss_mask_struct,
            )

        elif (self.data_mode == "smiles_struct_avh") and (
            self.process_mode == "autoregressive"
        ):
            # FH: A bit more complicated because you have additional sequences under consideration, such as the atomic number, the valency,
            #   and the hybridization. Not the most elegant, but separating out this case for now
            sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)
            smi_tokens = torch.tensor(self.smi_tokens[idx], dtype=torch.long)
            struct_tokens = torch.tensor(self.structure_tokens[idx], dtype=torch.long)
            masked_smi_tokens = torch.clone(smi_tokens)
            masked_struct_tokens = torch.clone(struct_tokens)

            # Format the input smiles sequence for cross entropy loss
            struct_token_indices = masked_smi_tokens == self.structure_token
            masked_smi_tokens[struct_token_indices.nonzero().max()] = (
                self.smiles_pad_token
            )
            # FH: More generally, set the last structure token to also be padding
            masked_struct_tokens[0][
                (masked_struct_tokens[0] != self.structure_pad_token).nonzero().max()
            ] = self.structure_pad_token

            # Format the target sequence for cross entropy
            smi_tokens = torch.roll(smi_tokens, shifts=-1, dims=0)
            smi_tokens[-1] = self.smiles_pad_token

            # The loss mask here should select out smiles tokens only
            loss_mask_smi = (smi_tokens != self.smiles_pad_token) & (
                smi_tokens != self.structure_token
            )
            # The structure loss mask should select against padding and masking tokens, where the masking tokens
            #   are used to enforce the spacing between the structural tokens and the three other parallel sequences
            loss_mask_struct = (struct_tokens[0] != self.structure_pad_token) & (
                struct_tokens[0] != self.structure_mask_token
            )

            return (
                self.data_mode,
                smi_tokens,
                masked_smi_tokens,
                struct_tokens,
                masked_struct_tokens,
                sequence_id,
                loss_mask_smi,
                loss_mask_struct,
            )


class GeometricTransformerDatasetAutoreg(Dataset):
    "Autoregressive variant of the GeometricTransformerDataset that does not apply masking"

    def __init__(self, get_hdf5_data: Callable, data_in_memory: bool = True) -> None:
        self.get_hdf5_data = get_hdf5_data
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5["smi_tokens"])
        self.mask_token_smi = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_MASK"][()]
        self.start_token_smi = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_START"][()]
        self.stop_token_smi = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_STOP"][()]
        del t_hdf5

    def __len__(self) -> int:
        return self.length

    def open_hdf5(self) -> None:
        self.t_hdf5 = self.get_hdf5_data()

        self.structural_tokens = self.t_hdf5["structural_tokens"]
        self.src_tokens = self.t_hdf5["smi_tokens"]
        self.tgt_tokens = self.t_hdf5["targets"]
        self.sequence_id = self.t_hdf5["sequence_id"]

        if self.data_in_memory:
            self.structural_tokens = self.structural_tokens[:]
            self.src_tokens = self.src_tokens[:]
            self.tgt_tokens = self.tgt_tokens[:]
            self.sequence_id = self.sequence_id[:]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        if not hasattr(self, "t_hdf5"):
            self.open_hdf5()

        structural_tokens = torch.tensor(self.structural_tokens[idx], dtype=torch.long)

        src_tokens = torch.tensor(self.src_tokens[idx], dtype=torch.long)
        src_key_pad_mask = (
            src_tokens == self.t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]
        )

        tgt_tokens = torch.tensor(self.tgt_tokens[idx], dtype=torch.long)
        if tgt_tokens.ndim == 2:
            # Assumption that leading dimension is 1
            tgt_tokens = tgt_tokens.squeeze(0)
        tgt_key_pad_mask = (
            tgt_tokens == self.t_hdf5["target_metadata"]["ctrl_tokens"]["TOK_PAD"][()]
        )

        sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)

        return (
            structural_tokens,
            src_tokens,
            src_key_pad_mask,
            tgt_tokens,
            tgt_key_pad_mask,
            sequence_id,
        )


class TransfusionDataset(Dataset):
    """
    Transfusion-based approach described in https://www.arxiv.org/abs/2408.11039

    The format of the transfusion dataset is linked to its precompute, as follows:
        - token_input: tokenized input sequence that takes into account the sections where structural information is incorporated
        - structure: structural information that is incorporated into the input sequence. Right now, this involves only the molecule's
            dihedral angles and a value of -100 is used for padding vectors to the same length within the dataset
        - smiles: SMILES string of the molecules
        - alphabet: alphabet used for tokenization
        - sequence_id: sequence id that indicates the type of token (structural, SMILES, padding)
        - input_metadata: control token sequences for the input

    The training of the transfusion model is done against the information contained in the input

    To featurize the input sequence, the dihedral angles are featurized as a sequence of sin and cosines over the angles
    """

    def __init__(
        self,
        get_hdf5_data: Callable,
        data_in_memory: bool = True,
        angle_feat: str = "sin_cos",
    ) -> None:
        assert angle_feat in ["sin_cos", "raw"]
        self.get_hdf5_data = get_hdf5_data
        self.data_in_memory = data_in_memory
        self.angle_feat = angle_feat
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5["token_input"])
        del t_hdf5

    def __len__(self) -> int:
        return self.length

    def open_hdf5(self) -> None:
        self.t_hdf5 = self.get_hdf5_data()

        self.structural_values = self.t_hdf5["structure"]
        self.tokenized_inputs = self.t_hdf5["token_input"]
        self.sequence_id = self.t_hdf5["sequence_id"]

        if self.data_in_memory:
            self.structural_values = self.structural_values[:]
            self.tokenized_inputs = self.tokenized_inputs[:]
            self.sequence_id = self.sequence_id[:]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        if not hasattr(self, "t_hdf5"):
            self.open_hdf5()

        # This sequence has the following form:
        #   <TOK_START><SMILES><STRUCT_START><STRUCT><STRUCT_END>
        token_input = torch.tensor(self.tokenized_inputs[idx], dtype=torch.long)
        # Shift over by one to get the target for cross entropy loss, removes stop token and the
        #   sequence id will automatically selects the <STRUCT_START> token for computing loss
        token_target = token_input[1:]
        structure_segment = torch.tensor(self.structural_values[idx], dtype=torch.float)
        sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)

        structure_padding_index = self.t_hdf5["input_metadata"]["ctrl_tokens"][
            "STRUCT_PAD"
        ][()]
        # Structure mask to exclude padding values for dihedrals
        struct_mask = structure_segment == structure_padding_index

        # Pre-featurize the dihedral angles
        if self.angle_feat == "sin_cos":
            # Since sin(pi) = 0 and cos(pi/2) = 0, we can use the structure mask to replace the padding values
            #   so that the transformation maps padding values to 0
            sine_pre = torch.tensor(structure_segment)
            sine_pre[struct_mask] = torch.pi
            cos_pre = torch.tensor(structure_segment)
            cos_pre[struct_mask] = torch.pi / 2
            sines = torch.sin(structure_segment)
            cosines = torch.cos(structure_segment)
            structure_segment = torch.stack([sines, cosines], dim=-1)

        return (token_input, token_target, structure_segment, struct_mask, sequence_id)


class TransformerERADataset(Dataset):
    def __init__(self, get_hdf5_data, data_in_memory=True, pair_selection="fixed"):
        """
        Dataset for using pre-specified alignment information
        """
        self.get_hdf5_data = get_hdf5_data
        data_ptr = self.get_hdf5_data()
        self.load_to_memory = data_in_memory
        self.num_pred_per_prompt = data_ptr["num_pred_per_prompt"][()]
        self.num_pairs_per_prompt = (
            self.num_pred_per_prompt * (self.num_pred_per_prompt - 1) // 2
        )
        self.num_prompts = data_ptr["num_prompts"][()]
        self.length = int(self.num_prompts * self.num_pairs_per_prompt)
        self.all_pairs = [
            [i, j]
            for i in range(self.num_pred_per_prompt)
            for j in range(i + 1, self.num_pred_per_prompt)
        ]
        del data_ptr
        assert pair_selection in ["fixed", "random"]
        self.pair_selection = pair_selection

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.data_hdf5 = self.get_hdf5_data()
        self.energies = self.data_hdf5["energies"]
        self.ref_logps = self.data_hdf5["ref_logps"]
        self.tokens = self.data_hdf5["tokens"]
        self.masks = self.data_hdf5["masks"]
        self.data_opened = True

        if self.load_to_memory:
            self.energies = self.energies[()]
            self.ref_logps = self.ref_logps[()]
            self.tokens = self.tokens[()]
            self.masks = self.masks[()]

    def __getitem__(self, idx):
        if not hasattr(self, "data_opened"):
            self.open_hdf5()
        if self.pair_selection == "fixed":
            prompt_idx = idx // self.num_pairs_per_prompt
            pair_idx = idx % self.num_pairs_per_prompt
            pair = self.all_pairs[pair_idx]
            idx1 = prompt_idx * self.num_pred_per_prompt + pair[0]
            idx2 = prompt_idx * self.num_pred_per_prompt + pair[1]
        elif self.pair_selection == "random":
            idx1 = torch.randint(0, self.tokens.shape[0] - 2, (1,)).item()
            idx2 = torch.randint(idx1 + 1, self.tokens.shape[0] - 1, (1,)).item()
        assert idx1 != idx2
        tokens = torch.tensor(self.tokens[[idx1, idx2]], dtype=torch.long)
        masks = torch.tensor(self.masks[[idx1, idx2]], dtype=torch.long)
        energies = torch.tensor(self.energies[[idx1, idx2]], dtype=torch.float)
        ref_logps = torch.tensor(self.ref_logps[[idx1, idx2]], dtype=torch.float)
        return (tokens, masks, energies, ref_logps)


class TransformerDPODataset(Dataset):
    def __init__(self, get_hdf5_data, data_in_memory=True, pair_selection="fixed"):
        """
        Dataset for using pre-specified alignment information
        """
        self.get_hdf5_data = get_hdf5_data
        data_ptr = self.get_hdf5_data()
        self.load_to_memory = data_in_memory
        self.contrastive_threshold = data_ptr["threshold_split_index"][()]
        num_pairs = (
            data_ptr["tokens"].shape[0] - self.contrastive_threshold
        ) * self.contrastive_threshold
        self.length = int(num_pairs)
        del data_ptr

    def __len__(self):
        return self.length

    def open_hdf5(self):
        self.data_hdf5 = self.get_hdf5_data()
        self.energies = self.data_hdf5["energies"]
        self.ref_logps = self.data_hdf5["ref_logps"]
        self.tokens = self.data_hdf5["tokens"]
        self.masks = self.data_hdf5["masks"]
        self.data_opened = True

        if self.load_to_memory:
            self.energies = self.energies[()]
            self.ref_logps = self.ref_logps[()]
            self.tokens = self.tokens[()]
            self.masks = self.masks[()]

    def __getitem__(self, idx):
        if not hasattr(self, "data_opened"):
            self.open_hdf5()
        idx1 = torch.randint(0, self.contrastive_threshold - 1, (1,)).item()
        idx2 = torch.randint(
            self.contrastive_threshold,
            self.tokens.shape[0] - 1,
            (1,),
        ).item()
        assert idx1 != idx2
        tokens = torch.tensor(self.tokens[[idx1, idx2]], dtype=torch.long)
        masks = torch.tensor(self.masks[[idx1, idx2]], dtype=torch.long)
        energies = torch.tensor(self.energies[[idx1, idx2]], dtype=torch.float)
        ref_logps = torch.tensor(self.ref_logps[[idx1, idx2]], dtype=torch.float)
        return (tokens, masks, energies, ref_logps)


class TransformerOnlineDataset(Dataset):
    def __init__(self, get_hdf5_data, data_in_memory=True):
        """
        Dataset for using pre-specified alignment information
        """
        self.get_hdf5_data = get_hdf5_data
        self.load_to_memory = data_in_memory

    def __len__(self):
        if not hasattr(self, "data_opened"):
            self.open_hdf5()
        return self.tokens.shape[0]

    def open_hdf5(self):
        self.data_hdf5 = self.get_hdf5_data()
        self.tokens = self.data_hdf5["tokens"]
        self.masks = self.data_hdf5["masks"]
        self.data_opened = True

        if self.load_to_memory:
            self.tokens = self.tokens[()]
            self.masks = self.masks[()]

    def __getitem__(self, idx):
        if not hasattr(self, "data_opened"):
            self.open_hdf5()
        tokens = torch.tensor(self.tokens[idx], dtype=torch.long)
        masks = torch.tensor(self.masks[idx], dtype=torch.long)
        return (tokens, masks)


def transformer_sharded_autoreg_collate_fn(
    data: tuple[torch.Tensor],
) -> tuple[torch.Tensor]:
    (
        structural_tokens,
        src_tokens,
        src_key_pad_mask,
        tgt_tokens,
        tgt_key_pad_mask,
        sequence_id,
        pad_tokens,
    ) = zip(*data)
    # Now need to be careful about padding everything to the same size
    pad_tokens = pad_tokens[0]
    smi_pad_token, struct_pad_token = pad_tokens

    # Using rnn padding, presumably works to generate a correct batch
    structural_tokens = pad_sequence(
        structural_tokens, batch_first=True, padding_value=struct_pad_token
    )
    src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=smi_pad_token)
    src_key_pad_mask = pad_sequence(
        src_key_pad_mask, batch_first=True, padding_value=True
    )
    tgt_tokens = pad_sequence(tgt_tokens, batch_first=True, padding_value=smi_pad_token)
    tgt_key_pad_mask = pad_sequence(
        tgt_key_pad_mask, batch_first=True, padding_value=True
    )
    sequence_id = pad_sequence(sequence_id, batch_first=True, padding_value=0)

    return (
        structural_tokens,
        src_tokens,
        src_key_pad_mask,
        tgt_tokens,
        tgt_key_pad_mask,
        sequence_id,
    )


class TransformerShardedAutoregDataset(Dataset):
    """
    Dataset that reads from multiple h5 shards when training the model autoregressively

    This dataset does not support loading data into memory as all shards will be interacted
    with via h5py pointers
    """

    def __init__(self, get_hdf5_data: Callable, data_in_memory: bool = False) -> None:
        self.get_hdf5_data = get_hdf5_data  # This should return a list of hdf5 pointers
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        # For this dataset, it should be a list of hdf5 pointers
        assert isinstance(t_hdf5, list)
        self._metadata_check(t_hdf5)

        ref_ptr = t_hdf5[0]
        smi_token_pad = (
            ref_ptr["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]
            if "TOK_PAD" in ref_ptr["input_metadata"]["ctrl_tokens"].keys()
            else None
        )
        assert smi_token_pad is not None, "No padding token found in metadata"
        struct_token_pad = (
            ref_ptr["input_metadata"]["ctrl_tokens"]["STRUCT_PAD"][()]
            if "STRUCT_PAD" in ref_ptr["input_metadata"]["ctrl_tokens"].keys()
            else None
        )
        if struct_token_pad is None:
            # Set to same as smiles padding token in nonetype case
            struct_token_pad = smi_token_pad
        # Smiles padding token first, structure padding second
        self.pad_tokens = [smi_token_pad, struct_token_pad]

        self.all_lengths = [len(ptr["smi_tokens"]) for ptr in t_hdf5]
        self.total_length = sum(self.all_lengths)
        self.len_cum_sum = torch.cumsum(torch.tensor(self.all_lengths), 0)

        print("Total number of datasets:", len(t_hdf5))
        print("Total dataset length:", self.total_length)
        print("Cumulative summation of lengths:", self.len_cum_sum)
        print("SMILES and structural pad tokens:", self.pad_tokens)

        del t_hdf5

    def __len__(self) -> int:
        return self.total_length

    def _unpack_metadata(self, metadata_obj):
        """
        Unpacks a specific metadata object into a Python dictionary

        The only keys we care about for consistency are the control tokens and the source size
        """
        metadata_dict = {}
        metadata_dict["control_tokens"] = {}
        for key in metadata_obj["ctrl_tokens"].keys():
            metadata_dict["control_tokens"][key] = metadata_obj["ctrl_tokens"][key][()]
        if "source_size" in metadata_obj.keys():
            metadata_dict["source_size"] = metadata_obj["source_size"][()]
        elif "target_size" in metadata_obj.keys():
            metadata_dict["target_size"] = metadata_obj["target_size"][()]
        return metadata_dict

    def _metadata_check(self, t_hdf5: list):
        """
        Given a list of hdf5 pointers, checks that each dataset was constructed
        using the same token metadata (padding, mask, start, stop, etc.)

        If something doesn't match, raise an error
        """
        input_metadatas_ref = None
        target_metadatas_red = None
        for i_ptr, ptr in enumerate(t_hdf5):
            if "input_metadata" in ptr.keys():
                input_metadata = self._unpack_metadata(ptr["input_metadata"])
                if input_metadatas_ref is None:
                    input_metadatas_ref = input_metadata
                else:
                    assert input_metadata == input_metadatas_ref, (
                        f"Input metadata mismatch at index {i_ptr}"
                    )
            if "target_metadata" in ptr.keys():
                target_metadata = self._unpack_metadata(ptr["target_metadata"])
                if target_metadatas_red is None:
                    target_metadatas_red = target_metadata
                else:
                    assert target_metadata == target_metadatas_red, (
                        f"Target metadata mismatch at index {i_ptr}"
                    )
        assert (
            input_metadatas_ref["control_tokens"]
            == target_metadatas_red["control_tokens"]
        ), "Control token mismatch between input and target"

    def open_hdf5(self) -> None:
        self.t_hdf5 = self.get_hdf5_data()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        if not hasattr(self, "t_hdf5"):
            self.open_hdf5()

        # Find the shard that contains the index
        diff = idx - self.len_cum_sum
        current_shard_idx = torch.where(diff < 0)[0][0]
        if current_shard_idx > 0:
            idx = idx - self.len_cum_sum[current_shard_idx - 1]
        # Get the data from the shard
        current_shard = self.t_hdf5[current_shard_idx]
        structural_tokens = torch.tensor(
            current_shard["structural_tokens"][idx], dtype=torch.long
        )
        src_tokens = torch.tensor(current_shard["smi_tokens"][idx], dtype=torch.long)
        src_key_pad_mask = src_tokens == self.pad_tokens[0]
        tgt_tokens = torch.tensor(current_shard["targets"][idx], dtype=torch.long)
        if tgt_tokens.ndim == 2:
            tgt_tokens = tgt_tokens.squeeze(0)
        tgt_key_pad_mask = tgt_tokens == self.pad_tokens[0]
        sequence_id = torch.tensor(current_shard["sequence_id"][idx], dtype=torch.long)

        return (
            structural_tokens,
            src_tokens,
            src_key_pad_mask,
            tgt_tokens,
            tgt_key_pad_mask,
            sequence_id,
            self.pad_tokens,
        )


def ida_transformer_mixed_seq_collate_fn(
    data: tuple[torch.Tensor],
) -> tuple[torch.Tensor]:
    zipped_data = tuple(zip(*data))
    mode = zipped_data[0][0]
    if mode == "ida":
        (
            _,
            smi_tokens,
            masked_smi_tokens,
            struct_tokens,
            masked_struct_tokens,
            coords,
            sequence_id,
            loss_mask_smi,
            loss_mask_struct,
        ) = zipped_data
        smi_tokens = torch.stack(smi_tokens)
        masked_smi_tokens = torch.stack(masked_smi_tokens)
        struct_tokens = torch.stack(struct_tokens)
        masked_struct_tokens = torch.stack(masked_struct_tokens)
        coords = torch.stack(coords)
        sequence_id = torch.stack(sequence_id)
        loss_mask_smi = torch.stack(loss_mask_smi)
        loss_mask_struct = torch.stack(loss_mask_struct)
        return (
            mode,
            smi_tokens,
            masked_smi_tokens,
            struct_tokens,
            masked_struct_tokens,
            coords,
            sequence_id,
            loss_mask_smi,
            loss_mask_struct,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}. Expected 'ida'.")


class IDATransformerDataset(Dataset):
    def __init__(
        self,
        get_hdf5_data: Callable,
        data_in_memory: bool = True,
        data_mode: str = "smiles_and_struct",
        process_mode: str = "masked",
        fixed_masking_on_end_tokens: bool = True,
    ) -> None:
        """
        get_hdf5_data: Callable, a function that returns the hdf5 data pointer
        data_in_memory: bool, whether to load the data into memory
        data_mode: str, one of ['smiles_only', 'smiles_and_struct']
        process_mode: str, one of ['masked', 'autoreg'] where 'masked' is bidirectional modeling
        fixed_masking_on_end_tokens: bool, whether to always mask the end tokens. Default is True
        """
        assert data_mode in [
            "ida",
        ]
        self.get_hdf5_data = get_hdf5_data
        self.data_mode = data_mode
        self.process_mode = process_mode
        self.fixed_masking_on_end_tokens = fixed_masking_on_end_tokens
        self.data_in_memory = data_in_memory
        t_hdf5 = self.get_hdf5_data()
        self.length = len(t_hdf5["smi_tokens"])
        # Unpack some useful metadata
        self.smiles_pad_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]
        self.smiles_start_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_START"][
            ()
        ]
        self.smiles_stop_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_STOP"][()]
        self.smiles_mask_token = t_hdf5["input_metadata"]["ctrl_tokens"]["TOK_MASK"][()]

        self.structure_token = t_hdf5["input_metadata"]["ctrl_tokens"]["STRUCT"][()]
        self.structure_pad_token = t_hdf5["input_metadata"]["ctrl_tokens"][
            "STRUCT_PAD"
        ][()]
        self.structure_start_token = t_hdf5["input_metadata"]["ctrl_tokens"][
            "STRUCT_START"
        ][()]
        self.structure_stop_token = t_hdf5["input_metadata"]["ctrl_tokens"][
            "STRUCT_STOP"
        ][()]
        self.structure_mask_token = t_hdf5["input_metadata"]["ctrl_tokens"][
            "STRUCT_MASK"
        ][()]

        # get the coordinate padding token
        # TODO: currently not used, maybe necessary for masking logic
        # self.coords_pad_token = t_hdf5["input_metadata"]["ctrl_tokens"][
        #    "COORDS_PAD"
        # ][()]

        del t_hdf5

    def __len__(self) -> int:
        return self.length

    def open_hdf5(self) -> None:
        self.t_hdf5 = self.get_hdf5_data()

        self.structure_tokens = self.t_hdf5["structure_tokens"]
        self.coords = self.t_hdf5["coords"] if "coords" in self.t_hdf5 else None
        self.smi_tokens = self.t_hdf5["smi_tokens"]
        self.sequence_id = self.t_hdf5["sequence_id"]

        if self.data_in_memory:
            # GR TODO: handle the cases where no structure tokens are present?
            self.structure_tokens = self.structure_tokens[:]
            if hasattr(self, "coords") and self.coords is not None:
                self.coords = self.coords[:]
            self.smi_tokens = self.smi_tokens[:]
            self.sequence_id = self.sequence_id[:]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        if not hasattr(self, "t_hdf5"):
            self.open_hdf5()

        if (self.data_mode == "ida") and (self.process_mode == "autoregressive"):
            # InteratomicDistanceAttention mode: includes 3D coordinates along with AVH tokens and structure tokens
            sequence_id = torch.tensor(self.sequence_id[idx], dtype=torch.long)
            smi_tokens = torch.tensor(self.smi_tokens[idx], dtype=torch.long)
            struct_tokens = torch.tensor(self.structure_tokens[idx], dtype=torch.long)

            # Load coordinates if available
            if hasattr(self, "coords") and self.coords is not None:
                coords = torch.tensor(self.coords[idx], dtype=torch.float32)
            else:
                coords = None

            # Apply same preprocessing as smiles_struct_avh
            masked_smi_tokens = torch.clone(smi_tokens)
            masked_struct_tokens = torch.clone(struct_tokens)

            # Format the input smiles sequence for cross entropy loss
            struct_token_indices = masked_smi_tokens == self.structure_token
            if struct_token_indices.any():
                masked_smi_tokens[struct_token_indices.nonzero().max()] = (
                    self.smiles_pad_token
                )

            # Set the last structure token to padding
            if struct_tokens.ndim > 1 and struct_tokens.shape[0] > 0:
                non_pad_indices = (
                    masked_struct_tokens[0] != self.structure_pad_token
                ).nonzero()
                if len(non_pad_indices) > 0:
                    masked_struct_tokens[0][non_pad_indices.max()] = (
                        self.structure_pad_token
                    )

            # TODO: Coordinate masking logic can be added here if needed

            # Format the target sequence for cross entropy
            smi_tokens = torch.roll(smi_tokens, shifts=-1, dims=0)
            smi_tokens[-1] = self.smiles_pad_token

            # Loss masks
            loss_mask_smi = (smi_tokens != self.smiles_pad_token) & (
                smi_tokens != self.structure_token
            )
            loss_mask_struct = (struct_tokens[0] != self.structure_pad_token) & (
                struct_tokens[0] != self.structure_mask_token
            )
            # GR coords not used in loss calculation, so don't need loss mask

            return (
                self.data_mode,
                smi_tokens,
                masked_smi_tokens,
                struct_tokens,
                masked_struct_tokens,
                coords,
                sequence_id,
                loss_mask_smi,
                loss_mask_struct,
            )
