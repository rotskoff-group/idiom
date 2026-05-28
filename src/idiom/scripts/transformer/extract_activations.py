import json
import os
import re
import tempfile
import time
from datetime import timedelta
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
    if dataset_filename.endswith((".fasta", ".fa")):
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
    struct_tokens = np.full(processed_inputs.shape, input_pad)

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


# Slab the activation dataset is chunked by. Picked to be roughly 4096*896*4 ≈ 14 MB
# per chunk — large enough that sequential slab reads are one or two chunks,
# small enough that h5py's per-chunk overhead stays negligible.
DATA_CHUNK_ROWS = 4096


def _get_ctrl_token_ids(token_info) -> torch.Tensor:
    """Token ids for PAD/START/STOP/MASK — never saved as activation rows."""
    return torch.tensor(
        [int(v) for k, v in token_info["input"]["TOK"].items() if k != "TOK_MAX_SIZE"],
        dtype=torch.long,
    )


def _get_fim_token_ids(token_info) -> torch.Tensor:
    """Token ids for FIM markers '1'/'3'/'2'.

    Looked up against the alphabet stored on the precompute tempfile (which is
    the same alphabet the tokenizer uses, since ``ResiduesInputBasic`` builds
    ``index_map = {char: i for i, char in enumerate(alphabet)}``). Raises if any
    marker is missing — that would indicate a non-FIM extraction and the caller
    almost certainly didn't mean to drop those rows.
    """
    alphabet = token_info.get("alphabet", None)
    if alphabet is None:
        raise RuntimeError(
            "No alphabet in token_info; cannot identify FIM marker token ids."
        )
    alpha_list = [
        a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
        for a in alphabet
    ]
    try:
        ids = [alpha_list.index("1"), alpha_list.index("3"), alpha_list.index("2")]
    except ValueError as e:
        raise RuntimeError(
            f"FIM marker missing from alphabet {alpha_list!r}: {e}"
        ) from e
    return torch.tensor(ids, dtype=torch.long)


def _write_shard_output(
    output_path,
    layers,
    d_model,
    np_dtype,
    alphabet,
    per_layer_arrays,
    seq_strings,
):
    """Write the final per-shard HDF5 in one shot with pre-shuffled rows.

    ``per_layer_arrays`` maps ``layer_idx`` → ``(data, seq_idx, pos_idx)`` where
    the three arrays have already been jointly permuted (so reading any
    contiguous range of rows gives a uniform random sample of the shard).
    Activation rows for FIM markers are NOT in these arrays; ``pos_idx`` still
    indexes positions in ``sequences/strings[seq_idx]`` (the raw FIM string is
    unchanged), so the row → (sequence, position) mapping is preserved.
    """
    with h5py.File(output_path, "w") as hf:
        meta = hf.create_group("metadata")
        meta.create_dataset("layers", data=np.array(layers, dtype=np.int32))
        if alphabet is not None:
            meta.create_dataset("alphabet", data=alphabet)

        for layer_idx in layers:
            data, seq_idx, pos_idx = per_layer_arrays[layer_idx]
            grp = hf.create_group(f"activations/layer_{layer_idx}")
            n = int(data.shape[0])
            data_chunks = (min(DATA_CHUNK_ROWS, max(n, 1)), d_model)
            grp.create_dataset(
                "data",
                data=data,
                dtype=np_dtype,
                chunks=data_chunks,
            )
            idx_chunks = (min(65536, max(n, 1)),)
            grp.create_dataset(
                "seq_idx",
                data=seq_idx,
                dtype=np.int32,
                chunks=idx_chunks,
            )
            grp.create_dataset(
                "pos_idx",
                data=pos_idx,
                dtype=np.int32,
                chunks=idx_chunks,
            )

        hf.create_dataset(
            "sequences/strings",
            data=np.array(seq_strings, dtype=object),
            dtype=h5py.string_dtype(),
        )


def _extract_shard(
    *,
    lightning_model,
    device,
    shard_idx,
    seq_start,
    seq_end,
    batch_size,
    ctrl_token_ids,
    skip_token_ids,
    d_model,
    alphabet,
    dataset_filename,
    layers,
    save_dtype,
    output_dir,
    shuffle_seed,
):
    """Extract one shard's sequence range and write its HDF5 file.

    Activation rows for control tokens (PAD/START/STOP/MASK) AND for the FIM
    markers '1'/'3'/'2' are dropped at write time. Within each shard the
    surviving rows are jointly permuted using a deterministic per-shard seed,
    so any sequential read of the shard is a uniform random sample of its
    residue tokens.

    Returns a dict of per-layer row counts (used by the manifest writer).
    """
    dataloader = _build_dataloader(dataset_filename, batch_size, seq_start, seq_end)
    np_dtype = np.float16 if save_dtype == "float16" else np.float32
    torch_dtype = torch.float16 if save_dtype == "float16" else torch.float32
    output_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.h5")

    per_layer_chunks: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    seq_idx_chunks: list[np.ndarray] = []
    pos_idx_chunks: list[np.ndarray] = []
    seq_strings: list[str] = []

    res_fh = h5py.File(dataset_filename, "r")
    try:
        global_seq_offset = seq_start
        for batch in dataloader:
            structural_tokens, src_tokens, _, _, _, sequence_id = batch
            B = src_tokens.shape[0]

            with torch.no_grad():
                _, hidden_states = lightning_model.model(
                    src_tokens.to(device),
                    structural_tokens.to(device),
                    sequence_id.to(device),
                    return_hidden_states=True,
                )

            # valid_mask: tokens kept by the model (not PAD/START/STOP/MASK) —
            #             gives the FIM-string position index via its cumulative index.
            # residue_mask: valid AND not a FIM marker — the rows we save.
            valid_mask_np = (~torch.isin(src_tokens, ctrl_token_ids)).numpy()
            residue_mask_np = (~torch.isin(src_tokens, skip_token_ids)).numpy()

            per_seq_residue_pos: list[np.ndarray] = []
            for b in range(B):
                valid_pos = np.where(valid_mask_np[b])[0]
                residue_pos = np.where(residue_mask_np[b])[0]
                per_seq_residue_pos.append(residue_pos)

                if len(residue_pos) > 0:
                    # pos in the raw FIM string = index within the kept-token
                    # (valid_pos) sequence. valid_pos is strictly increasing,
                    # so np.searchsorted gives that index in O(K).
                    pos_in_fim = np.searchsorted(valid_pos, residue_pos).astype(
                        np.int32
                    )
                    seq_idx_chunks.append(
                        np.full(len(residue_pos), global_seq_offset + b, dtype=np.int32)
                    )
                    pos_idx_chunks.append(pos_in_fim)

                raw = res_fh["residues"][global_seq_offset + b]
                seq_strings.append(
                    raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                )

            for layer_idx in layers:
                hs = hidden_states[layer_idx].cpu().to(torch_dtype).numpy()  # [B,L,D]
                for b in range(B):
                    residue_pos = per_seq_residue_pos[b]
                    if len(residue_pos) == 0:
                        continue
                    per_layer_chunks[layer_idx].append(hs[b, residue_pos])

            global_seq_offset += B
    finally:
        res_fh.close()

    # Concat across all batches; permute jointly with a per-shard RNG.
    if seq_idx_chunks:
        seq_idx_all = np.concatenate(seq_idx_chunks)
        pos_idx_all = np.concatenate(pos_idx_chunks)
    else:
        seq_idx_all = np.zeros(0, dtype=np.int32)
        pos_idx_all = np.zeros(0, dtype=np.int32)
    n_rows = len(seq_idx_all)

    rng = np.random.default_rng(int(shuffle_seed) + int(shard_idx))
    perm = rng.permutation(n_rows) if n_rows > 0 else np.zeros(0, dtype=np.int64)

    per_layer_arrays: dict[int, tuple] = {}
    rows_per_layer: dict[int, int] = {}
    for layer_idx in layers:
        if per_layer_chunks[layer_idx]:
            data_all = np.concatenate(per_layer_chunks[layer_idx], axis=0)
        else:
            data_all = np.zeros((0, d_model), dtype=np_dtype)
        assert len(data_all) == n_rows, (
            f"layer {layer_idx}: data rows {len(data_all)} != index rows {n_rows}"
        )
        if n_rows > 0:
            data_all = data_all[perm]
        per_layer_arrays[layer_idx] = (data_all, seq_idx_all[perm], pos_idx_all[perm])
        rows_per_layer[layer_idx] = int(len(data_all))
        per_layer_chunks[layer_idx] = []  # drop ref ASAP — these are large

    _write_shard_output(
        output_path=output_path,
        layers=layers,
        d_model=d_model,
        np_dtype=np_dtype,
        alphabet=alphabet,
        per_layer_arrays=per_layer_arrays,
        seq_strings=seq_strings,
    )
    return rows_per_layer


def _run_gpu_worker(
    gpu_id,
    shard_indices,
    shard_ranges,
    batch_size,
    model_args,
    training_args,
    token_info,
    d_model,
    alphabet,
    dataset_filename,
    layers,
    save_dtype,
    output_dir,
    checkpoint_path,
    shuffle_seed,
    result_queue,
):
    """Entry point for one GPU process. Loads the model once, processes its shards."""
    try:
        device = f"cuda:{gpu_id}"
        lightning_model = LightningModel(
            model_args=model_args, token_info=token_info, training_args=training_args
        )
        lightning_model.load_model_from_checkpoint(checkpoint_path)
        lightning_model.model.eval()
        lightning_model.to(device)

        ctrl_token_ids = _get_ctrl_token_ids(token_info)
        fim_token_ids = _get_fim_token_ids(token_info)
        # Union of ids that must NOT appear as activation rows. Control tokens
        # are already stripped from the model input as pads; FIM markers are
        # valid input tokens we explicitly drop from the saved activations.
        skip_token_ids = torch.unique(torch.cat([ctrl_token_ids, fim_token_ids]))

        for shard_idx in shard_indices:
            seq_start, seq_end = shard_ranges[shard_idx]
            rows = _extract_shard(
                lightning_model=lightning_model,
                device=device,
                shard_idx=shard_idx,
                seq_start=seq_start,
                seq_end=seq_end,
                batch_size=batch_size,
                ctrl_token_ids=ctrl_token_ids,
                skip_token_ids=skip_token_ids,
                d_model=d_model,
                alphabet=alphabet,
                dataset_filename=dataset_filename,
                layers=layers,
                save_dtype=save_dtype,
                output_dir=output_dir,
                shuffle_seed=shuffle_seed,
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


def _partition(n: int, k: int) -> list[tuple[int, int]]:
    """Split ``[0, n)`` into ``k`` contiguous half-open ranges.

    Excess (when ``n % k != 0``) goes to the first ``n % k`` ranges, each
    getting one extra element. Used for both the sequence→shard split and the
    shard→GPU split — contiguous-block assignment keeps tempfile reads
    sequential per worker.
    """
    base, rem = divmod(n, k)
    ranges = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def _write_manifest(
    output_dir, total_sequences, layers, save_dtype, alphabet, shard_results
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

        d_model = model_args["model_args"]["d_model"]
        alphabet = token_info.get("alphabet", None)

        shard_ranges = _partition(total_sequences, num_shards)
        # GPU g owns the contiguous block of shard indices [s, e).
        gpu_assignment = [
            list(range(s, e)) for s, e in _partition(num_shards, num_gpus)
        ]

        print(f"Extracting {total_sequences:,} sequences into {num_shards} shard(s)")
        print(f"Layers: {layers}, dtype: {save_dtype}")

        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
        OmegaConf.save(cfg, os.path.join(output_dir, "extract_config.yaml"))

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        result_queue: "mp.Queue" = mp.Queue()
        processes = []
        t_start_all = time.monotonic()
        shuffle_seed = int(training_args["seed_args"]["seed"])
        for gpu_id, shard_indices in enumerate(gpu_assignment):
            if not shard_indices:
                continue
            p = mp.Process(
                target=_run_gpu_worker,
                args=(
                    gpu_id,
                    shard_indices,
                    shard_ranges,
                    batch_size,
                    model_args,
                    training_args,
                    token_info,
                    d_model,
                    alphabet,
                    temp_shard_path,
                    layers,
                    save_dtype,
                    output_dir,
                    checkpoint_path,
                    shuffle_seed,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

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
            elapsed = timedelta(seconds=int(time.monotonic() - t_start_all))
            print(f"{done}/{expected} shards done — elapsed {elapsed}", flush=True)

        for p in processes:
            p.join()

        if errors:
            for e in errors:
                print(f"GPU {e.get('gpu_id', '?')} error: {e['error']}", flush=True)
            raise RuntimeError(f"{len(errors)} shard worker(s) failed; see logs above")

        _write_manifest(
            output_dir, total_sequences, layers, save_dtype, alphabet, shard_results
        )

    finally:
        if os.path.exists(temp_shard_path):
            os.remove(temp_shard_path)


if __name__ == "__main__":
    main()
