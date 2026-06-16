"""On-the-fly training dataset (Option A): records -> FIM -> tokenize -> shifted targets.

No precompute / token-h5. Each ``__getitem__`` assembles a FIM string from a :class:`Record`
(``full`` vs ``132`` chosen at random per sample — the augmentation that replaces the old
stored ×2 duplication), tokenizes it, and forms the next-token-prediction pair:

    input  = [START, t0, t1, ..., t_{n-1}]
    target = [t0,    t1, ..., t_{n-1}, STOP]

(same length, shifted by one). Padding is added in :func:`make_collate`, and the loss
ignores ``pad_id``.

This is map-style over an in-memory record list — simple and fully CPU-testable, sufficient
for the de-risk-gate small-model run. Scaling to the full ~37M corpus uses a sharded/streaming
variant that reuses :func:`record_to_example` (tracked in P1/P3).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from loguru import logger as log
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from idiom.data.fim import fim_132, fim_full
from idiom.data.io import Record
from idiom.data.record_store import RecordStore
from idiom.data.tokenizer import Tokenizer

# A full example is START + ``1{prefix}3{suffix}2{IDR}``: the 3 FIM markers + START over the
# residues. So model positions = len(full_seq) + 4. Records longer than that are dropped.
FIM_OVERHEAD = 4


def max_protein_len(max_len: int) -> int:
    """Largest ``full_seq`` length whose ``full`` example fits in ``max_len`` positions."""
    return max_len - FIM_OVERHEAD


def record_to_example(
    record: Record, tokenizer: Tokenizer, *, variant: str = "full"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(input_ids, target_ids)`` next-token pair for one record + FIM variant."""
    build = fim_full if variant == "full" else fim_132
    ids = tokenizer.encode(build(record.full_seq, record.idr_start, record.idr_end))
    input_ids = torch.tensor([tokenizer.start_id, *ids], dtype=torch.long)
    target_ids = torch.tensor([*ids, tokenizer.stop_id], dtype=torch.long)
    return input_ids, target_ids


class RecordDataset(Dataset):
    """Map-style dataset over :class:`Record`s with on-the-fly FIM + tokenization."""

    def __init__(
        self,
        records,
        tokenizer: Tokenizer | None = None,
        *,
        max_len: int = 1024,
        fim_full_prob: float = 0.5,
        completion_only: bool = False,
        seed: int = 0,
    ) -> None:
        self.tok = tokenizer or Tokenizer()
        self.max_len = int(max_len)
        self.fim_full_prob = float(fim_full_prob)
        # completion_only=True -> SFT: compute loss only on the IDR completion (after the `2`
        # marker). False -> pretraining: loss on every token.
        self.completion_only = bool(completion_only)
        self._rng = random.Random(seed)

        keep = max_protein_len(self.max_len)  # filter on the full-context length so any sample fits
        if isinstance(records, RecordStore):
            # Memory-mapped store: never materialize Records; keep an index of the rows that fit.
            self.store: RecordStore | None = records
            self.records = None
            self._keep = np.nonzero(records.seq_lengths() <= keep)[0]
            n_total = len(records)
            n_kept = len(self._keep)
        else:
            self.store = None
            records = list(records)  # materialize (may be a generator) so we can filter + index
            self.records = [r for r in records if len(r.full_seq) <= keep]
            self._keep = None
            n_total = len(records)
            n_kept = len(self.records)
        if n_total - n_kept:
            log.info(f"RecordDataset: dropped {n_total - n_kept} record(s) longer than max_len={self.max_len}")

    def __len__(self) -> int:
        return len(self._keep) if self.store is not None else len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.store[int(self._keep[i])] if self.store is not None else self.records[i]
        variant = "full" if self._rng.random() < self.fim_full_prob else "132"  # per-sample augmentation
        x, y = record_to_example(rec, self.tok, variant=variant)
        if self.completion_only:
            # The IDR is the trailing part of the FIM string, so target's last (idr_len + 1)
            # positions are the IDR residues + STOP — the completion to train on for SFT.
            idr_len = rec.idr_end - rec.idr_start
            mask = torch.zeros(y.size(0), dtype=torch.bool)
            mask[-(idr_len + 1) :] = True
        else:
            mask = torch.ones(y.size(0), dtype=torch.bool)  # pretraining: loss on all tokens
        return x, y, mask


def make_collate(pad_id: int):
    """Collate ``(input, target, loss_mask)`` triples into right-padded ``[B, L]`` batches."""

    def collate(batch):
        inputs, targets, masks = zip(*batch)
        x = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        y = pad_sequence(targets, batch_first=True, padding_value=pad_id)
        m = pad_sequence(masks, batch_first=True, padding_value=False)  # pad positions: no loss
        return x, y, m

    return collate
