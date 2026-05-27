import os
import re
import tempfile
import time
import h5py
import numpy as np
import torch
import hydra
import lightning as L
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from idiom.nn.transformer.module import LightningModel
from idiom.nn.transformer.dataset import (
    TransformerShardedAutoregDataset,
    transformer_sharded_autoreg_collate_fn,
)
from idiom.utils.token import aggregate_tokens_hdf5
from idiom.nn.transformer.utils.tokenizer import CharTokenizer
from idiom.nn.transformer.generators.input_generators import ResiduesInputBasic
from idiom.nn.transformer.generators.target_generators import ResiduesTarget
from idiom.scripts.transformer.precompute import (
    determine_alphabet,
    run_process_parallel,
)

_IDR_PATTERN = re.compile(r"_IDR_(\d+)-(\d+)")


def _fmt_hms(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--:--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def _parse_fasta_to_fim(fasta_path: str, max_sequences: int | None) -> list[str]:
    """Parse a FASTA file with _IDR_x-y headers and return FIM-formatted sequences.

    Headers must contain _IDR_x-y (1-indexed, inclusive) to define the IDR region.
    FIM format: 1{prefix}3{suffix}2{IDR}
    Sequences without a valid _IDR_ annotation are skipped with a warning.
    """
    sequences = []
    header = None
    seq_parts = []

    def _flush():
        if header is None:
            return
        m = _IDR_PATTERN.search(header)
        if m is None:
            print(f"Warning: no _IDR_x-y found in header '{header}', skipping")
            return
        idr_start_1 = int(m.group(1))
        idr_end_1 = int(m.group(2))
        seq = "".join(seq_parts)
        idr_start_0 = idr_start_1 - 1
        idr_end_0 = idr_end_1 - 1
        prefix = seq[:idr_start_0]
        idr = seq[idr_start_0 : idr_end_0 + 1]
        suffix = seq[idr_end_0 + 1 :]
        sequences.append(f"1{prefix}3{suffix}2{idr}")

    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                header = line[1:]
                seq_parts = []
            elif line:
                seq_parts.append(line)
    _flush()

    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    print(f"Parsed {len(sequences):,} FIM-formatted sequences from {fasta_path}")
    return sequences


def _load_residues(dataset_filename: str, max_sequences: int | None) -> list[str]:
    """Load FIM-formatted residue strings.

    Accepts either:
      - a FASTA (.fasta/.fa) with ``_IDR_x-y`` headers — converted to FIM format
        ``1{prefix}3{suffix}2{IDR}`` here, or
      - an HDF5 file with a ``residues`` field whose strings are already in
        FIM format.
    """
    if dataset_filename.endswith(".fasta") or dataset_filename.endswith(".fa"):
        return _parse_fasta_to_fim(dataset_filename, max_sequences)
    with h5py.File(dataset_filename, "r") as f:
        total = len(f["residues"])
        n = min(total, max_sequences) if max_sequences is not None else total
        # "residues" dataset must be stored as bytes (utf-8 encoded)
        assert h5py.check_string_dtype(f["residues"].dtype).encoding == "utf-8", (
            f"Expected 'residues' dataset to be utf-8 encoded bytes, "
            f"got dtype {f['residues'].dtype}"
        )
        return [f["residues"][i].decode("utf-8") for i in range(n)]


def _precompute_to_tempfile(
    residues: list[str],
    source_label: str,
    num_workers: int,
) -> str:
    """Tokenize residue strings into a temporary shard HDF5 and return its path.

    The caller is responsible for deleting the file when done.
    """
    print(f"Tokenizing {len(residues):,} sequences from {source_label}...")

    tokenizer = CharTokenizer()
    alphabet = determine_alphabet(residues, tokenizer)

    input_gen = ResiduesInputBasic(
        residues, tokenizer, alphabet, apply_start=True, apply_stop=False
    )
    target_gen = ResiduesTarget(
        residues, tokenizer, alphabet, targets=None, apply_start=False, apply_stop=True
    )

    processed_inputs = run_process_parallel(
        input_gen.transform, {}, num_workers, residues
    )
    # ResiduesTarget.transform ignores its second argument; pass dummy values so
    # run_process_parallel can zip the two data sequences together.
    processed_targets = run_process_parallel(
        target_gen.transform, {}, num_workers, residues, [None] * len(residues)
    )

    processed_inputs = np.array([np.array(x) for x in processed_inputs])
    processed_targets = np.array([np.array(x) for x in processed_targets])

    input_pad = input_gen.get_ctrl_tokens()["TOK_PAD"]
    sequence_id = (processed_inputs != input_pad).astype(processed_inputs.dtype)
    struct_tokens = np.ones(processed_inputs.shape) * input_pad

    fd, temp_path = tempfile.mkstemp(suffix=".h5", prefix="idiom_extract_")
    os.close(fd)

    with h5py.File(temp_path, "w") as f:
        f.create_dataset("res_tokens", data=processed_inputs)
        f.create_dataset("targets", data=processed_targets)
        f.create_dataset("residues", data=residues)
        f.create_dataset("alphabet", data=alphabet)
        f.create_dataset("sequence_id", data=sequence_id)
        f.create_dataset("structural_tokens", data=struct_tokens)

        inp_meta = f.create_group("input_metadata")
        inp_meta.create_dataset("source_size", data=input_gen.get_size())
        inp_meta.create_dataset("max_seq_len", data=input_gen.get_max_seq_len())
        ctrl = inp_meta.create_group("ctrl_tokens")
        for k, v in input_gen.get_ctrl_tokens().items():
            ctrl.create_dataset(k, data=v)

        tar_meta = f.create_group("target_metadata")
        tar_meta.create_dataset("target_size", data=target_gen.get_size())
        tar_meta.create_dataset("max_seq_len", data=target_gen.get_max_seq_len())
        ctrl = tar_meta.create_group("ctrl_tokens")
        for k, v in target_gen.get_ctrl_tokens().items():
            ctrl.create_dataset(k, data=v)

    print(f"Precompute complete: {len(residues):,} sequences → {temp_path}")
    return temp_path


def _load_model(model_args, training_args, token_info, checkpoint_path, device):
    """Build LightningModel, load checkpoint, set eval, move to device."""
    lightning_model = LightningModel(
        model_args=model_args, token_info=token_info, training_args=training_args
    )
    lightning_model.load_model_from_checkpoint(checkpoint_path)
    lightning_model.model.eval()
    lightning_model.to(device)
    return lightning_model


def _build_dataloader(dataset_filename, batch_size):
    """DataLoader over every sequence in the precomputed shard."""

    def get_hdf5():
        return [h5py.File(dataset_filename, "r")]

    dataset = TransformerShardedAutoregDataset(get_hdf5_data=get_hdf5)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=transformer_sharded_autoreg_collate_fn,
        num_workers=0,
    )


def _get_ctrl_token_ids(token_info) -> torch.Tensor:
    """Token ids for PAD/START/STOP/MASK — the only ids filtered from saved rows."""
    return torch.tensor(
        [int(v) for k, v in token_info["input"]["TOK"].items() if k != "TOK_MAX_SIZE"],
        dtype=torch.long,
    )


def _init_output_h5(hf, layers, d_model, np_dtype, alphabet):
    """Create the canonical output layout and return per-layer + per-sequence handles."""
    meta = hf.create_group("metadata")
    meta.create_dataset("layers", data=np.array(layers, dtype=np.int32))
    if alphabet is not None:
        meta.create_dataset("alphabet", data=alphabet)

    layer_grps = {}
    for layer_idx in layers:
        grp = hf.create_group(f"activations/layer_{layer_idx}")
        grp.create_dataset(
            "data",
            shape=(0, d_model),
            maxshape=(None, d_model),
            dtype=np_dtype,
            chunks=(256, d_model),
        )
        grp.create_dataset(
            "seq_idx",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(65536,),
        )
        grp.create_dataset(
            "pos_idx",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(65536,),
        )
        layer_grps[layer_idx] = grp

    vlen_int = h5py.vlen_dtype(np.dtype("int16"))
    tokens_ds = hf.create_dataset(
        "sequences/tokens",
        shape=(0,),
        maxshape=(None,),
        dtype=vlen_int,
        chunks=(4096,),
    )
    strings_ds = hf.create_dataset(
        "sequences/strings",
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(),
        chunks=(4096,),
    )
    return layer_grps, tokens_ds, strings_ds


def _extract(
    batch_size,
    model_args,
    training_args,
    token_info,
    dataset_filename,
    layers,
    save_dtype,
    output_path,
    checkpoint_path,
):
    device = "cuda:0"
    lightning_model = _load_model(
        model_args, training_args, token_info, checkpoint_path, device
    )
    dataloader = _build_dataloader(dataset_filename, batch_size)

    d_model = model_args["model_args"]["d_model"]
    np_dtype = np.float16 if save_dtype == "float16" else np.float32
    torch_dtype = torch.float16 if save_dtype == "float16" else torch.float32

    # Only PAD/START/STOP/MASK are filtered from saved rows — FIM markers
    # ('1', '3', '2') are kept like any other residue token.
    ctrl_token_ids = _get_ctrl_token_ids(token_info)
    print(f"ctrl_token_ids={ctrl_token_ids.tolist()}")

    alphabet = token_info.get("alphabet", None)
    total_sequences = len(dataloader.dataset)

    res_fh = h5py.File(dataset_filename, "r")
    try:
        with h5py.File(output_path, "w") as hf:
            layer_grps, tokens_ds, strings_ds = _init_output_h5(
                hf, layers, d_model, np_dtype, alphabet
            )

            global_seq_offset = 0
            t_start = time.monotonic()
            seq_at_last_log = 0
            log_every = batch_size * 10
            print(f"Starting extraction over {total_sequences:,} sequences")
            for batch in dataloader:
                (
                    structural_tokens,
                    src_tokens,
                    src_key_pad_mask,
                    _,
                    _,
                    sequence_id,
                ) = batch
                B = src_tokens.shape[0]

                with torch.no_grad():
                    _, hidden_states = lightning_model.model(
                        src_tokens.to(device),
                        structural_tokens.to(device),
                        sequence_id.to(device),
                        return_hidden_states=True,
                    )

                # True iff not PAD/START/STOP/MASK. FIM markers '1'/'3'/'2'
                # pass through and are saved alongside residues.
                valid_mask = ~torch.isin(src_tokens, ctrl_token_ids)
                valid_mask_np = valid_mask.numpy()

                # Precompute per-row metadata once and share across layers.
                seq_idx_chunks, pos_idx_chunks, per_seq_valid_pos = [], [], []
                for b in range(B):
                    valid_pos = np.where(valid_mask_np[b])[0]
                    per_seq_valid_pos.append(valid_pos)
                    if len(valid_pos) == 0:
                        continue
                    seq_idx_chunks.append(
                        np.full(len(valid_pos), global_seq_offset + b, dtype=np.int32)
                    )
                    pos_idx_chunks.append(np.arange(len(valid_pos), dtype=np.int32))

                if seq_idx_chunks:
                    seq_idx_all = np.concatenate(seq_idx_chunks)
                    pos_idx_all = np.concatenate(pos_idx_chunks)
                else:
                    seq_idx_all = np.zeros(0, dtype=np.int32)
                    pos_idx_all = np.zeros(0, dtype=np.int32)

                for layer_idx in layers:
                    hs = (
                        hidden_states[layer_idx].cpu().to(torch_dtype).numpy()
                    )  # [B, L, D]
                    grp = layer_grps[layer_idx]

                    act_chunks = []
                    for b in range(B):
                        valid_pos = per_seq_valid_pos[b]
                        if len(valid_pos) == 0:
                            continue
                        act_chunks.append(hs[b, valid_pos])

                    if act_chunks:
                        act_all = np.concatenate(act_chunks, axis=0)
                        old = grp["data"].shape[0]
                        new = old + len(act_all)
                        grp["data"].resize(new, axis=0)
                        grp["data"][old:new] = act_all
                        grp["seq_idx"].resize(new, axis=0)
                        grp["seq_idx"][old:new] = seq_idx_all
                        grp["pos_idx"].resize(new, axis=0)
                        grp["pos_idx"][old:new] = pos_idx_all

                # Free GPU memory held by hidden_states before the next forward pass
                del hidden_states
                torch.cuda.empty_cache()

                # Per-sequence kept token ids + raw FIM-formatted residue string.
                old = tokens_ds.shape[0]
                tokens_ds.resize(old + B, axis=0)
                strings_ds.resize(old + B, axis=0)
                for b in range(B):
                    valid_pos = per_seq_valid_pos[b]
                    kept_ids = src_tokens[b, valid_pos].numpy()
                    tokens_ds[old + b] = kept_ids.astype(np.int16)
                    raw = res_fh["residues"][global_seq_offset + b]
                    strings_ds[old + b] = (
                        raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    )

                global_seq_offset += B

                if (
                    global_seq_offset - seq_at_last_log >= log_every
                    or global_seq_offset == total_sequences
                ):
                    elapsed = time.monotonic() - t_start
                    rate = global_seq_offset / max(elapsed, 1e-9)
                    eta = (
                        (total_sequences - global_seq_offset) / rate
                        if rate > 0
                        else float("inf")
                    )
                    print(
                        f"[{_fmt_hms(elapsed)}] {global_seq_offset:,}/{total_sequences:,} "
                        f"({100 * global_seq_offset / total_sequences:.1f}%)  ETA {_fmt_hms(eta)}",
                        flush=True,
                    )
                    seq_at_last_log = global_seq_offset
    finally:
        res_fh.close()

    print(
        f"Activations saved to {output_path} (took {_fmt_hms(time.monotonic() - t_start)})"
    )


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="extract")
def main(cfg) -> None:
    model_args = cfg["model"]
    training_args = cfg["training"]
    extract_args = cfg["extract"]

    L.seed_everything(**training_args["seed_args"])

    checkpoint_path = extract_args["checkpoint_path"]
    dataset_filename = extract_args["dataset_filename"]
    output_path = extract_args["output_path"]
    layers = list(extract_args["layers"])
    # Validate layers against the model's transformer depth
    n_layers = model_args["model_args"]["unified_transformer_args"]["n_layers"]
    if not layers:
        raise ValueError("extract.layers must be non-empty")
    if len(set(layers)) != len(layers):
        raise ValueError(f"extract.layers contains duplicates: {layers}")
    out_of_range = [li for li in layers if not (0 <= li < n_layers)]
    if out_of_range:
        raise ValueError(
            f"extract.layers contains out-of-range indices {out_of_range}; "
            f"valid range for this {n_layers}-layer model is 0..{n_layers - 1}"
        )
    batch_size = extract_args["batch_size"]
    max_sequences = extract_args.get("max_sequences", None)
    save_dtype = extract_args.get("save_dtype", "float16")
    num_precompute_workers = extract_args.get("num_precompute_workers", 4)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for activation extraction")

    residues = _load_residues(dataset_filename, max_sequences)
    temp_shard_path = _precompute_to_tempfile(
        residues, dataset_filename, num_precompute_workers
    )

    try:
        with h5py.File(temp_shard_path, "r") as f:
            token_info = aggregate_tokens_hdf5(f)
            total_sequences = len(f["res_tokens"])
        print(f"Extracting activations for {total_sequences:,} sequences")
        print(f"Layers: {layers}, dtype: {save_dtype}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        savedir = os.path.dirname(os.path.abspath(output_path))
        OmegaConf.save(cfg, os.path.join(savedir, "extract_config.yaml"))

        _extract(
            batch_size,
            model_args,
            training_args,
            token_info,
            temp_shard_path,
            layers,
            save_dtype,
            output_path,
            checkpoint_path,
        )

    finally:
        if os.path.exists(temp_shard_path):
            os.remove(temp_shard_path)
            print(f"Removed temporary shard: {temp_shard_path}")


if __name__ == "__main__":
    main()
