import torch
from torch.utils.data import Dataset
from typing import Callable
from torch.nn.utils.rnn import pad_sequence


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
    res_pad_token, struct_pad_token = pad_tokens

    # Using rnn padding, presumably works to generate a correct batch
    structural_tokens = pad_sequence(
        structural_tokens, batch_first=True, padding_value=struct_pad_token
    )
    src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=res_pad_token)
    src_key_pad_mask = pad_sequence(
        src_key_pad_mask, batch_first=True, padding_value=True
    )
    tgt_tokens = pad_sequence(tgt_tokens, batch_first=True, padding_value=res_pad_token)
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


def transformer_online_collate_fn(data):
    (tokens, masks) = zip(*data)

    tokens = torch.stack(tokens, dim=0)
    masks = torch.stack(masks, dim=0)

    return (tokens, masks)


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
        res_token_pad = (
            ref_ptr["input_metadata"]["ctrl_tokens"]["TOK_PAD"][()]
            if "TOK_PAD" in ref_ptr["input_metadata"]["ctrl_tokens"].keys()
            else None
        )
        assert res_token_pad is not None, "No padding token found in metadata"
        struct_token_pad = (
            ref_ptr["input_metadata"]["ctrl_tokens"]["STRUCT_PAD"][()]
            if "STRUCT_PAD" in ref_ptr["input_metadata"]["ctrl_tokens"].keys()
            else None
        )
        if struct_token_pad is None:
            # Set to same as residues padding token in nonetype case
            struct_token_pad = res_token_pad
        # Residues padding token first, structure padding second
        self.pad_tokens = [res_token_pad, struct_token_pad]

        self.all_lengths = [len(ptr["res_tokens"]) for ptr in t_hdf5]
        self.total_length = sum(self.all_lengths)
        self.len_cum_sum = torch.cumsum(torch.tensor(self.all_lengths), 0)

        print("Total number of datasets:", len(t_hdf5))
        print("Total dataset length:", self.total_length)
        print("Cumulative summation of lengths:", self.len_cum_sum)
        print("Residues and structural pad tokens:", self.pad_tokens)

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
        src_tokens = torch.tensor(current_shard["res_tokens"][idx], dtype=torch.long)
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
