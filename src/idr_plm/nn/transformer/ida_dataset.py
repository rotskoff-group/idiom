import torch
from torch.utils.data import Dataset
from typing import Callable


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
            coords,
            sequence_id,
            loss_mask_smi,
            loss_mask_struct,
        ) = zipped_data
        smi_tokens = torch.stack(smi_tokens)
        masked_smi_tokens = torch.stack(masked_smi_tokens)
        struct_tokens = torch.stack(struct_tokens)
        coords = torch.stack(coords)
        sequence_id = torch.stack(sequence_id)
        loss_mask_smi = torch.stack(loss_mask_smi)
        loss_mask_struct = torch.stack(loss_mask_struct)
        return (
            mode,
            smi_tokens,
            masked_smi_tokens,
            struct_tokens,
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
                coords,
                sequence_id,
                loss_mask_smi,
                loss_mask_struct,
            )
