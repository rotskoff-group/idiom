import random

import h5py
import torch
import torch.nn.functional as F


def load_token_info_from_shard(shard_path):
    """
    Load alphabet and special token indices from a precomputed HDF5 shard.

    Returns a dict with keys:
        alphabet     - list of character strings
        char_to_idx  - dict mapping character -> token index
        start_token  - int
        stop_token   - int
        pad_token    - int
    """
    with h5py.File(shard_path, "r") as f:
        alphabet = [x.decode("utf-8") for x in f["alphabet"][()]]
        start_token = int(f["input_metadata/ctrl_tokens/TOK_START"][()])
        stop_token = int(f["input_metadata/ctrl_tokens/TOK_STOP"][()])
        pad_token = int(f["input_metadata/ctrl_tokens/TOK_PAD"][()])
    return {
        "alphabet": alphabet,
        "char_to_idx": {c: i for i, c in enumerate(alphabet)},
        "start_token": start_token,
        "stop_token": stop_token,
        "pad_token": pad_token,
    }


def _parse_fasta(fasta_path):
    header = None
    seq_lines = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        yield header, "".join(seq_lines)


def _fim_string(seq, idr_start_0, idr_end_0):
    """
    Build FIM string: 1{prefix}3{suffix}2{IDR}
    idr_start_0 and idr_end_0 are 0-indexed, end inclusive.
    Returns (fim_string, idr_start_in_fim) where idr_start_in_fim is the
    0-indexed position in fim_string where the IDR characters begin (after '2').
    """
    prefix = seq[:idr_start_0]
    middle = seq[idr_start_0 : idr_end_0 + 1]
    suffix = seq[idr_end_0 + 1 :]
    fim = f"1{prefix}3{suffix}2{middle}"
    idr_start_in_fim = len(prefix) + len(suffix) + 3  # 3 sentinel chars: '1', '3', '2'
    return fim, idr_start_in_fim


def compute_sequence_perplexity(
    model, fim_string, idr_start_in_fim, token_info, device
):
    """
    Compute perplexity for a single FIM-formatted sequence.

    Runs a single forward pass and returns both IDR-only and full-sequence perplexity.

    Args:
        model: LightningModel (only model.model is called)
        fim_string: Full FIM string "1{prefix}3{suffix}2{IDR}"
        idr_start_in_fim: 0-indexed position in fim_string where IDR chars begin
        token_info: dict from load_token_info_from_shard
        device: torch device

    Returns:
        (ppl_idr, ppl_full) tuple of floats, or None if the sequence contains
        unknown characters
    """
    char_to_idx = token_info["char_to_idx"]
    start_token = token_info["start_token"]
    stop_token = token_info["stop_token"]

    indices = []
    for c in fim_string:
        idx = char_to_idx.get(c)
        if idx is None:
            return None
        indices.append(idx)

    # full_tokens = [START, fim[0], ..., fim[N-1], STOP]
    full_tokens = torch.tensor(
        [start_token] + indices + [stop_token], dtype=torch.long, device=device
    )

    # Autoregressive: input = full[:-1], target = full[1:]
    input_tokens = full_tokens[:-1].unsqueeze(0)  # (1, T)
    target_tokens = full_tokens[1:].unsqueeze(0)  # (1, T)

    T = input_tokens.shape[1]
    seq_id = torch.ones(1, T, dtype=torch.long, device=device)
    structural = torch.zeros(1, T, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model.model(input_tokens, structural, seq_id)  # (1, T, vocab)

    # IDR-only: positions [idr_start_in_fim:] (IDR tokens + STOP)
    ppl_idr = torch.exp(
        F.cross_entropy(
            logits[0, idr_start_in_fim:, :],
            target_tokens[0, idr_start_in_fim:],
        )
    ).item()

    # Full sequence: all FIM tokens + STOP (positions [0:])
    ppl_full = torch.exp(
        F.cross_entropy(
            logits[0, :, :],
            target_tokens[0, :],
        )
    ).item()

    return ppl_idr, ppl_full


def compute_perplexity_from_fasta(
    model,
    fasta_path,
    shard_path,
    n_sample=1000,
    seed=42,
    device=None,
    full_sequence_as_idp=False,
):
    """
    Compute perplexity for a random sample of sequences from a FASTA file.

    Alphabet and special tokens are loaded from shard_path (an HDF5 precomputed shard),
    following the same pattern used elsewhere in the project.

    By default each header must contain '_IDR_x-y' (1-indexed, inclusive) and the
    sequence is scored in FIM format 1{prefix}3{suffix}2{IDR}.

    If full_sequence_as_idp=True the entire sequence is treated as a fully disordered
    protein regardless of the header: FIM = "132{sequence}" with no flanking context.
    This is useful for structured-protein datasets (e.g. CATH) used as a negative control.

    Args:
        model: LightningModel with loaded checkpoint
        fasta_path: Path to FASTA file
        shard_path: Path to precomputed HDF5 shard (for alphabet and token indices)
        n_sample: Number of sequences to randomly sample (default 1000)
        seed: Random seed (default 42)
        device: torch device (defaults to model's device)
        full_sequence_as_idp: If True, ignore IDR bounds and score the full sequence
            as "132{sequence}" (default False)

    Returns:
        list of (header, ppl_idr, ppl_full) tuples, skipping sequences with parse
        errors or unknown characters
    """
    if device is None:
        device = next(model.model.parameters()).device

    model.model.eval()
    token_info = load_token_info_from_shard(shard_path)

    records = list(_parse_fasta(fasta_path))

    rng = random.Random(seed)
    if len(records) > n_sample:
        records = rng.sample(records, n_sample)

    results = []
    for header, sequence in records:
        if full_sequence_as_idp:
            # "132{sequence}": sentinels occupy positions 0-2, IDR starts at 3
            fim = f"132{sequence}"
            idr_start_in_fim = 3
        else:
            if "_IDR_" not in header:
                continue

            _, idr_part = header.split("_IDR_", 1)
            try:
                start_str, end_str = idr_part.split("-", 1)
                idr_start_1 = int(start_str)
                idr_end_1 = int(end_str)
            except ValueError:
                continue

            idr_start_0 = idr_start_1 - 1
            idr_end_0 = idr_end_1 - 1

            if idr_end_0 >= len(sequence) or idr_start_0 < 0:
                continue

            fim, idr_start_in_fim = _fim_string(sequence, idr_start_0, idr_end_0)

        result = compute_sequence_perplexity(
            model, fim, idr_start_in_fim, token_info, device
        )
        if result is not None:
            ppl_idr, ppl_full = result
            results.append((header, ppl_idr, ppl_full))

    return results
