import os
import re
import tempfile
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import hydra
import lightning as L
from torch.utils.data import DataLoader, Subset
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
from idiom.utils.extract_helpers import (
    _alphabet_chars,
    _compute_fim_segments,
    _get_ctrl_token_ids,
    _get_filter_token_ids,
)

_IDR_PATTERN = re.compile(r"_IDR_(\d+)-(\d+)")


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
    """Load residue strings from an H5 file or a FASTA file with _IDR_ headers."""
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


def _build_dataloader(dataset_filename, start_idx, end_idx, batch_size):
    """DataLoader over [start_idx, end_idx) of the precomputed shard."""

    def get_hdf5():
        return [h5py.File(dataset_filename, "r")]

    dataset = TransformerShardedAutoregDataset(get_hdf5_data=get_hdf5)
    subset = Subset(dataset, range(start_idx, end_idx))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=transformer_sharded_autoreg_collate_fn,
        num_workers=0,
    )


def _init_output_h5(hf, layers, d_model, np_dtype):
    """Create per-layer activation groups + tokens/strings datasets in an open H5 file."""
    layer_grps = {}
    for layer_idx in layers:
        grp = hf.create_group(f"layer_{layer_idx}")
        grp.create_dataset(
            "data",
            shape=(0, d_model),
            maxshape=(None, d_model),
            dtype=np_dtype,
            chunks=(1024, d_model),
        )
        grp.create_dataset(
            "seq_idx",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(4096,),
        )
        grp.create_dataset(
            "pos_idx",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(4096,),
        )
        grp.create_dataset(
            "local_pos_idx",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(4096,),
        )
        grp.create_dataset(
            "fim_segment",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int8,
            chunks=(4096,),
        )
        layer_grps[layer_idx] = grp

    vlen_int = h5py.vlen_dtype(np.dtype("int16"))
    tokens_ds = hf.create_dataset(
        "tokens", shape=(0,), maxshape=(None,), dtype=vlen_int, chunks=(256,)
    )
    strings_ds = hf.create_dataset(
        "strings",
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(),
        chunks=(256,),
    )
    aligned_strings_ds = hf.create_dataset(
        "aligned_strings",
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(),
        chunks=(256,),
    )
    return layer_grps, tokens_ds, strings_ds, aligned_strings_ds


def _extract_on_gpu(
    gpu_id,
    start_idx,
    end_idx,
    batch_size,
    model_args,
    training_args,
    token_info,
    dataset_filename,
    layers,
    save_dtype,
    savedir,
    checkpoint_path,
):
    try:
        device = f"cuda:{gpu_id}"
        lightning_model = _load_model(
            model_args, training_args, token_info, checkpoint_path, device
        )
        dataloader = _build_dataloader(dataset_filename, start_idx, end_idx, batch_size)

        d_model = model_args["model_args"]["d_model"]
        np_dtype = np.float16 if save_dtype == "float16" else np.float32

        # PAD/START/STOP/MASK and FIM markers ('1','3','2') are all filtered from
        # saved activations/tokens. Residues are labeled by which FIM segment they
        # belong to (1=prefix, 3=suffix, 2=IDR).
        ctrl_token_ids = _get_ctrl_token_ids(token_info)
        filter_token_ids, fim_id_to_label = _get_filter_token_ids(token_info)
        print(
            f"ctrl_token_ids={ctrl_token_ids.tolist()} "
            f"fim_id_to_label={fim_id_to_label}"
        )
        alphabet_chars = _alphabet_chars(token_info)

        temp_path = os.path.join(savedir, f"gpu_{gpu_id}_temp.h5")
        with h5py.File(temp_path, "w") as hf:
            layer_grps, tokens_ds, strings_ds, aligned_strings_ds = _init_output_h5(
                hf, layers, d_model, np_dtype
            )

            global_seq_offset = start_idx
            n_processed = 0

            res_fh = h5py.File(dataset_filename, "r")
            try:
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

                    # True iff a residue token (i.e. not PAD/START/STOP/MASK and not a
                    # FIM marker '1'/'3'/'2'). The kept rows match `aligned_strings`.
                    valid_mask = ~torch.isin(src_tokens, filter_token_ids)
                    fim_segments = _compute_fim_segments(
                        src_tokens,
                        fim_id_to_label,
                        ctrl_token_ids,
                        global_seq_offset,
                    )

                    for layer_idx in layers:
                        hs = (
                            hidden_states[layer_idx]
                            .cpu()
                            .to(
                                torch.float16
                                if save_dtype == "float16"
                                else torch.float32
                            )
                            .numpy()
                        )  # [B, L, D]
                        grp = layer_grps[layer_idx]

                        (
                            act_chunks,
                            seq_idx_chunks,
                            pos_idx_chunks,
                            local_pos_chunks,
                            fim_seg_chunks,
                        ) = ([], [], [], [], [])
                        for b in range(B):
                            valid_pos = np.where(valid_mask[b].numpy())[0]
                            if len(valid_pos) == 0:
                                continue
                            act_chunks.append(hs[b, valid_pos])
                            seq_idx_chunks.append(
                                np.full(
                                    len(valid_pos),
                                    global_seq_offset + b,
                                    dtype=np.int32,
                                )
                            )
                            pos_idx_chunks.append(valid_pos.astype(np.int32))
                            local_pos_chunks.append(
                                np.arange(len(valid_pos), dtype=np.int32)
                            )
                            fim_seg_chunks.append(fim_segments[b, valid_pos])

                        if act_chunks:
                            act_all = np.concatenate(act_chunks, axis=0)
                            seq_idx_all = np.concatenate(seq_idx_chunks)
                            pos_idx_all = np.concatenate(pos_idx_chunks)
                            local_pos_all = np.concatenate(local_pos_chunks)
                            fim_seg_all = np.concatenate(fim_seg_chunks)
                            old = grp["data"].shape[0]
                            new = old + len(act_all)
                            grp["data"].resize(new, axis=0)
                            grp["data"][old:new] = act_all
                            grp["seq_idx"].resize(new, axis=0)
                            grp["seq_idx"][old:new] = seq_idx_all
                            grp["pos_idx"].resize(new, axis=0)
                            grp["pos_idx"][old:new] = pos_idx_all
                            grp["local_pos_idx"].resize(new, axis=0)
                            grp["local_pos_idx"][old:new] = local_pos_all
                            grp["fim_segment"].resize(new, axis=0)
                            grp["fim_segment"][old:new] = fim_seg_all

                    # Free GPU memory held by hidden_states before the next forward pass
                    del hidden_states
                    torch.cuda.empty_cache()

                    # Store kept token sequences, raw residue strings, and the
                    # aligned residue strings (one char per kept activation row).
                    old = tokens_ds.shape[0]
                    tokens_ds.resize(old + B, axis=0)
                    strings_ds.resize(old + B, axis=0)
                    aligned_strings_ds.resize(old + B, axis=0)
                    for b in range(B):
                        valid_pos = np.where(valid_mask[b].numpy())[0]
                        kept_ids = src_tokens[b, valid_pos].numpy()
                        tokens_ds[old + b] = kept_ids.astype(np.int16)
                        raw = res_fh["residues"][global_seq_offset + b]
                        strings_ds[old + b] = (
                            raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                        )
                        aligned_strings_ds[old + b] = "".join(
                            alphabet_chars[int(tid)] for tid in kept_ids
                        )

                    global_seq_offset += B
                    n_processed += B

                    if gpu_id == 0 and n_processed % (batch_size * 20) < batch_size:
                        total = end_idx - start_idx
                        print(
                            f"GPU {gpu_id}: {n_processed}/{total} sequences "
                            f"({100 * n_processed / total:.1f}%)"
                        )
            finally:
                res_fh.close()

        print(f"GPU {gpu_id}: Done — saved to {temp_path}")

    except Exception as e:
        import traceback

        print(f"GPU {gpu_id}: Error — {e}")
        traceback.print_exc()


def _merge_temp_files(savedir, num_gpus, output_path, layers, token_info):
    alphabet = token_info.get("alphabet", None)

    with h5py.File(output_path, "w") as out:
        # Metadata
        meta = out.create_group("metadata")
        if alphabet is not None:
            meta.create_dataset("alphabet", data=alphabet)
        meta.create_dataset("layers", data=np.array(layers, dtype=np.int32))

        # Determine dtype and d_model from first temp file
        first_temp = os.path.join(savedir, "gpu_0_temp.h5")
        with h5py.File(first_temp, "r") as f0:
            sample_dtype = f0[f"layer_{layers[0]}"]["data"].dtype

        # Pre-create resizable activation datasets
        act_grps = {}
        for layer_idx in layers:
            grp = out.create_group(f"activations/layer_{layer_idx}")
            with h5py.File(first_temp, "r") as f0:
                d_model = f0[f"layer_{layer_idx}"]["data"].shape[1]
            grp.create_dataset(
                "data",
                shape=(0, d_model),
                maxshape=(None, d_model),
                dtype=sample_dtype,
                chunks=(1024, d_model),
            )
            grp.create_dataset(
                "seq_idx", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(4096,)
            )
            grp.create_dataset(
                "pos_idx",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(4096,),
            )
            grp.create_dataset(
                "local_pos_idx",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(4096,),
            )
            grp.create_dataset(
                "fim_segment",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int8,
                chunks=(4096,),
            )
            act_grps[layer_idx] = grp

        vlen_int = h5py.vlen_dtype(np.dtype("int16"))
        tokens_ds = out.create_dataset(
            "sequences/tokens",
            shape=(0,),
            maxshape=(None,),
            dtype=vlen_int,
            chunks=(256,),
        )
        strings_ds = out.create_dataset(
            "sequences/strings",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(),
            chunks=(256,),
        )
        aligned_strings_ds = out.create_dataset(
            "sequences/aligned_strings",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(),
            chunks=(256,),
        )

        # Copy fixed-shape activation datasets in chunks (avoid loading all rows
        # into RAM at once) and batch-write vlen/string datasets in slices
        # (avoid one HDF5 call per sequence).
        row_chunk = 1_000_000
        str_chunk = 4096

        for gpu_id in range(num_gpus):
            temp_path = os.path.join(savedir, f"gpu_{gpu_id}_temp.h5")
            if not os.path.exists(temp_path):
                print(f"Warning: temp file for GPU {gpu_id} not found, skipping")
                continue
            with h5py.File(temp_path, "r") as tmp:
                for layer_idx in layers:
                    grp = act_grps[layer_idx]
                    src = tmp[f"layer_{layer_idx}"]

                    n_new = src["data"].shape[0]
                    if n_new == 0:
                        continue

                    old = grp["data"].shape[0]
                    grp["data"].resize(old + n_new, axis=0)
                    grp["seq_idx"].resize(old + n_new, axis=0)
                    grp["pos_idx"].resize(old + n_new, axis=0)
                    grp["local_pos_idx"].resize(old + n_new, axis=0)
                    grp["fim_segment"].resize(old + n_new, axis=0)
                    for start in range(0, n_new, row_chunk):
                        stop = min(start + row_chunk, n_new)
                        grp["data"][old + start : old + stop] = src["data"][start:stop]
                        grp["seq_idx"][old + start : old + stop] = src["seq_idx"][
                            start:stop
                        ]
                        grp["pos_idx"][old + start : old + stop] = src["pos_idx"][
                            start:stop
                        ]
                        grp["local_pos_idx"][old + start : old + stop] = src[
                            "local_pos_idx"
                        ][start:stop]
                        grp["fim_segment"][old + start : old + stop] = src[
                            "fim_segment"
                        ][start:stop]

                n_toks = tmp["tokens"].shape[0]
                if n_toks > 0:
                    old_t = tokens_ds.shape[0]
                    tokens_ds.resize(old_t + n_toks, axis=0)
                    strings_ds.resize(old_t + n_toks, axis=0)
                    aligned_strings_ds.resize(old_t + n_toks, axis=0)
                    for start in range(0, n_toks, str_chunk):
                        stop = min(start + str_chunk, n_toks)
                        tokens_ds[old_t + start : old_t + stop] = tmp["tokens"][
                            start:stop
                        ]
                        strings_ds[old_t + start : old_t + stop] = tmp["strings"][
                            start:stop
                        ]
                        aligned_strings_ds[old_t + start : old_t + stop] = tmp[
                            "aligned_strings"
                        ][start:stop]

            os.remove(temp_path)
            print(f"Merged GPU {gpu_id} temp file")

    print(f"Activations saved to {output_path}")


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

    num_gpus = torch.cuda.device_count()
    use_multi_gpu = extract_args.get("use_multi_gpu", False) and num_gpus > 1
    if not use_multi_gpu:
        num_gpus = 1
    print(f"Using {num_gpus} GPU(s)")

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

        sequences_per_gpu = total_sequences // num_gpus
        remainder = total_sequences % num_gpus

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        processes = []
        for gpu_id in range(num_gpus):
            extra = 1 if gpu_id < remainder else 0
            if gpu_id < remainder:
                start_idx = gpu_id * (sequences_per_gpu + 1)
            else:
                start_idx = (
                    remainder * (sequences_per_gpu + 1)
                    + (gpu_id - remainder) * sequences_per_gpu
                )
            end_idx = start_idx + sequences_per_gpu + extra
            print(f"GPU {gpu_id}: sequences {start_idx}–{end_idx - 1}")

            p = mp.Process(
                target=_extract_on_gpu,
                args=(
                    gpu_id,
                    start_idx,
                    end_idx,
                    batch_size,
                    model_args,
                    training_args,
                    token_info,
                    temp_shard_path,
                    layers,
                    save_dtype,
                    savedir,
                    checkpoint_path,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("All GPU processes complete, merging results...")
        _merge_temp_files(savedir, num_gpus, output_path, layers, token_info)

    finally:
        if os.path.exists(temp_shard_path):
            os.remove(temp_shard_path)
            print(f"Removed temporary shard: {temp_shard_path}")


if __name__ == "__main__":
    main()
