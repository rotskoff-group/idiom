"""Held-out perplexity under the FIM objective — pure model quality, independent of generation.

Reuses the exact training data pipeline (`RecordDataset` + FIM) and masked next-token loss, so the
reported NLL matches what training optimizes, on a held-out record FASTA (e.g. the test split).
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer


@torch.no_grad()
def perplexity(
    model, fasta: str, *, tokenizer: Tokenizer | None = None, max_len: int = 1024,
    fim_full_prob: float = 0.5, batch_size: int = 32, num_workers: int = 4,
    device: str = "cuda", max_records: int | None = None, seed: int = 0,
) -> dict[str, float]:
    """Mean per-token NLL (nats) and perplexity over `fasta` under the FIM loss (pad/mask ignored)."""
    tok = tokenizer or Tokenizer()
    records = read_records(fasta)
    if max_records is not None:
        records = itertools.islice(records, max_records)
    ds = RecordDataset(records, tok, max_len=max_len, fim_full_prob=fim_full_prob,
                       completion_only=False, seed=seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=make_collate(tok.pad_id))
    model.eval()
    total_nll, total_tok = 0.0, 0
    for x, y, mask in dl:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
        ).view_as(y)
        total_nll += float((ce * mask).sum())
        total_tok += int(mask.sum())
    mean_nll = total_nll / max(total_tok, 1)
    return {"nll": mean_nll, "perplexity": math.exp(mean_nll), "n_tokens": total_tok}
