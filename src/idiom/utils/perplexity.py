"""Held-out next-token NLL and perplexity over record FASTAs."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.records import read_records
from idiom.data.tokenizer import Tokenizer


@torch.no_grad()
def perplexity(
    model,
    fasta: str,
    *,
    tokenizer: Tokenizer | None = None,
    max_len: int = 1024,
    prompted_prob: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    max_records: int | None = None,
    seed: int = 0,
    completion_only: bool = False,
) -> dict[str, float]:
    """Compute mean per-token NLL and perplexity over a record FASTA.

    Padded and masked-out positions are excluded. The model is put in evaluation mode
    but is not moved to device or restored to its previous mode.

    Args:
        model: The model to evaluate, called as model(input_ids) -> logits.
        fasta: Path to the held-out record FASTA.
        tokenizer: Tokenizer; defaults to Tokenizer().
        max_len: Maximum model positions; longer records are dropped.
        prompted_prob: Probability that a sample uses the prompted variant.
        batch_size: Batch size for the evaluation dataloader.
        num_workers: DataLoader worker processes.
        device: Device to run the model on.
        max_records: If set, evaluate only the first this many records.
        seed: Seed for FIM variant selection.
        completion_only: If True, score only IDR residues and the final STOP token,
            excluding flanks and FIM markers. Defaults to the full-token objective.

    Returns:
        A dictionary with "nll" (mean token NLL in nats), "perplexity" (its
        exponential), and "n_tokens" (scored-token count). With no scored tokens,
        these values are 0, 1, and 0, respectively.
    """
    tok = tokenizer or Tokenizer()
    records = read_records(fasta)
    if max_records is not None:
        records = itertools.islice(records, max_records)
    ds = RecordDataset(
        records, tok, max_len=max_len, prompted_prob=prompted_prob, completion_only=completion_only, seed=seed
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=make_collate(tok.pad_id)
    )
    model.eval()
    total_nll, total_tok = 0.0, 0
    for x, y, mask in dl:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view_as(y)
        total_nll += float((ce * mask).sum())
        total_tok += int(mask.sum())
    mean_nll = total_nll / max(total_tok, 1)
    return {"nll": mean_nll, "perplexity": math.exp(mean_nll), "n_tokens": total_tok}
