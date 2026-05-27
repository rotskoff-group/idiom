import json
import os
import re
import tempfile
import time
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


def _build_dataloader(dataset_filename, batch_size, seq_start=None, seq_end=None):
    """DataLoader over a (sub)range of sequences in the precomputed tempfile.

    When ``seq_start`` / ``seq_end`` are given, wraps the underlying dataset in a
    ``Subset`` so only that half-open range is iterated. Used to scope one shard's
    work to its assigned sequences during multi-shard extraction.
    """

    def get_hdf5():
        return [h5py.File(dataset_filename, "r")]

    dataset = TransformerShardedAutoregDataset(get_hdf5_data=get_hdf5)
    if seq_start is not None or seq_end is not None:
        seq_start = 0 if seq_start is None else seq_start
        seq_end = len(dataset) if seq_end is None else seq_end
        dataset = Subset(dataset, range(seq_start, seq_end))
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


def _init_output_h5(
    hf, layers, d_model, np_dtype, alphabet, shard_idx, num_shards, seq_start, seq_end
):
    """Create the canonical per-shard output layout and return handles.

    Per-shard metadata identifies the shard's position in the wider sharded
    extraction so downstream readers can stitch shards without re-scanning.
    ``seq_idx`` written into the activation rows is GLOBAL across shards.
    """
    meta = hf.create_group("metadata")
    meta.create_dataset("layers", data=np.array(layers, dtype=np.int32))
    if alphabet is not None:
        meta.create_dataset("alphabet", data=alphabet)
    meta.create_dataset("shard_idx", data=np.int32(shard_idx))
    meta.create_dataset("num_shards", data=np.int32(num_shards))
    meta.create_dataset("seq_start", data=np.int64(seq_start))
    meta.create_dataset("seq_end", data=np.int64(seq_end))

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


def _extract_shard(
    lightning_model,
    gpu_id,
    shard_idx,
    num_shards,
    seq_start,
    seq_end,
    batch_size,
    model_args,
    token_info,
    dataset_filename,
    layers,
    save_dtype,
    output_dir,
):
    """Extract one shard's sequence range and write its HDF5 file.

    Returns a dict of per-layer row counts (used by the manifest writer).
    """
    device = f"cuda:{gpu_id}"
    dataloader = _build_dataloader(dataset_filename, batch_size, seq_start, seq_end)

    d_model = model_args["model_args"]["d_model"]
    np_dtype = np.float16 if save_dtype == "float16" else np.float32
    torch_dtype = torch.float16 if save_dtype == "float16" else torch.float32

    ctrl_token_ids = _get_ctrl_token_ids(token_info)
    alphabet = token_info.get("alphabet", None)

    output_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.h5")
    shard_total = seq_end - seq_start
    log_prefix = f"[gpu {gpu_id} shard {shard_idx:04d}/{num_shards}]"

    res_fh = h5py.File(dataset_filename, "r")
    rows_per_layer = {layer_idx: 0 for layer_idx in layers}
    try:
        with h5py.File(output_path, "w") as hf:
            layer_grps, tokens_ds, strings_ds = _init_output_h5(
                hf,
                layers,
                d_model,
                np_dtype,
                alphabet,
                shard_idx=shard_idx,
                num_shards=num_shards,
                seq_start=seq_start,
                seq_end=seq_end,
            )

            global_seq_offset = seq_start
            t_start = time.monotonic()
            print(
                f"{log_prefix} starting: seqs [{seq_start}, {seq_end}) = {shard_total:,}",
                flush=True,
            )
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
                        rows_per_layer[layer_idx] = new

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
    finally:
        res_fh.close()

    print(
        f"{log_prefix} done: {output_path} "
        f"(took {_fmt_hms(time.monotonic() - t_start)})",
        flush=True,
    )
    return rows_per_layer


def _run_gpu_worker(
    gpu_id,
    shard_indices,
    shard_ranges,
    num_shards,
    batch_size,
    model_args,
    training_args,
    token_info,
    dataset_filename,
    layers,
    save_dtype,
    output_dir,
    checkpoint_path,
    result_queue,
):
    """Entry point for one GPU process. Loads the model once, processes its shards."""
    try:
        device = f"cuda:{gpu_id}"
        lightning_model = _load_model(
            model_args, training_args, token_info, checkpoint_path, device
        )
        for shard_idx in shard_indices:
            seq_start, seq_end = shard_ranges[shard_idx]
            rows = _extract_shard(
                lightning_model,
                gpu_id,
                shard_idx,
                num_shards,
                seq_start,
                seq_end,
                batch_size,
                model_args,
                token_info,
                dataset_filename,
                layers,
                save_dtype,
                output_dir,
            )
            result_queue.put(
                {
                    "shard_idx": shard_idx,
                    "seq_start": seq_start,
                    "seq_end": seq_end,
                    "rows_per_layer": rows,
                }
            )
    except Exception as e:
        import traceback

        traceback.print_exc()
        result_queue.put({"error": str(e), "gpu_id": gpu_id})


def _compute_shard_ranges(total_sequences: int, num_shards: int):
    """Split ``[0, total_sequences)`` into ``num_shards`` contiguous half-open ranges.

    Excess (when ``total_sequences % num_shards != 0``) is distributed across
    the first ``remainder`` shards, each getting one extra sequence — matches
    the assignment policy the old multi-GPU code used.
    """
    base = total_sequences // num_shards
    remainder = total_sequences % num_shards
    ranges = []
    start = 0
    for k in range(num_shards):
        size = base + (1 if k < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def _assign_shards_to_gpus(num_shards: int, num_gpus: int):
    """Contiguous-block assignment: GPU g gets a slice of shard indices.

    Each GPU handles roughly ``num_shards // num_gpus`` shards in sequence
    order, which keeps tempfile reads roughly contiguous per GPU.
    """
    base = num_shards // num_gpus
    rem = num_shards % num_gpus
    assignment: list[list[int]] = []
    start = 0
    for g in range(num_gpus):
        size = base + (1 if g < rem else 0)
        assignment.append(list(range(start, start + size)))
        start += size
    return assignment


def _write_manifest(
    output_dir, cfg, total_sequences, layers, save_dtype, alphabet, shard_results
):
    """Write a JSON inventory of the sharded output for downstream readers."""
    alphabet_list = None
    if alphabet is not None:
        alphabet_list = [
            a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
            for a in alphabet
        ]
    shards_sorted = sorted(shard_results, key=lambda d: d["shard_idx"])
    manifest = {
        "num_shards": len(shards_sorted),
        "total_sequences": total_sequences,
        "layers": list(map(int, layers)),
        "save_dtype": save_dtype,
        "alphabet": alphabet_list,
        "shards": [
            {
                "shard_idx": int(s["shard_idx"]),
                "file": f"shard_{int(s['shard_idx']):04d}.h5",
                "seq_start": int(s["seq_start"]),
                "seq_end": int(s["seq_end"]),
                "rows_per_layer": {
                    int(k): int(v) for k, v in s["rows_per_layer"].items()
                },
            }
            for s in shards_sorted
        ],
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {path}")


@hydra.main(version_base="1.3", config_path="../cfgs", config_name="extract")
def main(cfg) -> None:
    model_args = cfg["model"]
    training_args = cfg["training"]
    extract_args = cfg["extract"]

    L.seed_everything(**training_args["seed_args"])

    checkpoint_path = extract_args["checkpoint_path"]
    dataset_filename = extract_args["dataset_filename"]
    output_dir = extract_args["output_dir"]
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
    num_shards = int(extract_args["num_shards"])
    if num_shards < 1:
        raise ValueError(f"extract.num_shards must be >= 1, got {num_shards}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for activation extraction")
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    # Resolve CPU budget for the precompute step from SLURM (or fall back).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    num_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    npw_cfg = extract_args.get("num_precompute_workers", None)
    num_precompute_workers = int(npw_cfg) if npw_cfg else num_cpus
    num_precompute_workers = min(num_precompute_workers, num_cpus)
    print(
        f"Using {num_precompute_workers} CPU workers for precompute (SLURM_CPUS_PER_TASK={slurm_cpus})"
    )

    # Multiple GPU subprocesses will open the tempfile concurrently for reading.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    residues = _load_residues(dataset_filename, max_sequences)
    temp_shard_path = _precompute_to_tempfile(
        residues, dataset_filename, num_precompute_workers
    )

    try:
        with h5py.File(temp_shard_path, "r") as f:
            token_info = aggregate_tokens_hdf5(f)
            total_sequences = len(f["res_tokens"])
        # Don't create more shards than there are sequences.
        if num_shards > total_sequences:
            print(
                f"num_shards={num_shards} > total_sequences={total_sequences}; "
                f"reducing to {total_sequences}"
            )
            num_shards = total_sequences

        shard_ranges = _compute_shard_ranges(total_sequences, num_shards)
        gpu_assignment = _assign_shards_to_gpus(num_shards, num_gpus)

        print(f"Extracting {total_sequences:,} sequences into {num_shards} shard(s)")
        print(f"Layers: {layers}, dtype: {save_dtype}")
        for g, shards in enumerate(gpu_assignment):
            if not shards:
                continue
            first, last = shards[0], shards[-1]
            r0 = shard_ranges[first][0]
            r1 = shard_ranges[last][1]
            print(
                f"  GPU {g}: shards [{first:04d}..{last:04d}] → seqs [{r0:,}..{r1:,})"
            )

        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
        OmegaConf.save(cfg, os.path.join(output_dir, "extract_config.yaml"))

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        result_queue: "mp.Queue" = mp.Queue()
        processes = []
        t_start_all = time.monotonic()
        for gpu_id, shard_indices in enumerate(gpu_assignment):
            if not shard_indices:
                continue
            p = mp.Process(
                target=_run_gpu_worker,
                args=(
                    gpu_id,
                    shard_indices,
                    shard_ranges,
                    num_shards,
                    batch_size,
                    model_args,
                    training_args,
                    token_info,
                    temp_shard_path,
                    layers,
                    save_dtype,
                    output_dir,
                    checkpoint_path,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        # Drain the queue concurrently so a slow consumer doesn't block producers.
        shard_results: list[dict] = []
        errors: list[dict] = []
        expected = num_shards
        while len(shard_results) + len(errors) < expected:
            msg = result_queue.get()
            if "error" in msg:
                errors.append(msg)
            else:
                shard_results.append(msg)
            done = len(shard_results) + len(errors)
            elapsed = time.monotonic() - t_start_all
            rate = done / max(elapsed, 1e-9)
            eta = (expected - done) / rate if rate > 0 else float("inf")
            print(
                f"[overall {_fmt_hms(elapsed)}] {done}/{expected} shards done  ETA {_fmt_hms(eta)}",
                flush=True,
            )

        for p in processes:
            p.join()

        if errors:
            for e in errors:
                print(f"GPU {e.get('gpu_id', '?')} error: {e['error']}", flush=True)
            raise RuntimeError(f"{len(errors)} shard worker(s) failed; see logs above")

        alphabet = token_info.get("alphabet", None)
        _write_manifest(
            output_dir,
            cfg,
            total_sequences,
            layers,
            save_dtype,
            alphabet,
            shard_results,
        )
        print(
            f"All shards complete in {_fmt_hms(time.monotonic() - t_start_all)} "
            f"→ {output_dir}"
        )

    finally:
        if os.path.exists(temp_shard_path):
            os.remove(temp_shard_path)
            print(f"Removed temporary shard: {temp_shard_path}")


if __name__ == "__main__":
    main()
