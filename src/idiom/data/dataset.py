"""Next-token training examples from prompted or unprompted FIM records.

Inputs prepend START; targets append STOP. make_collate adds right padding.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
from loguru import logger as log
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from idiom.data.fim import PROMPTED, UNPROMPTED, fim_prompted, fim_unprompted, normalize_mode
from idiom.data.record_store import RecordStore
from idiom.data.records import Record
from idiom.data.tokenizer import Tokenizer

# Each input needs START and three FIM markers in addition to its residues
FIM_OVERHEAD = 4


def max_protein_len(max_len: int) -> int:
    """Return max_len minus the START token and three FIM markers."""
    return max_len - FIM_OVERHEAD


def record_to_example(
    record: Record, tokenizer: Tokenizer, *, variant: str = PROMPTED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (input_ids, target_ids) next-token pair for one record.

    Args:
        record: The record to encode.
        tokenizer: Character tokenizer for the FIM string.
        variant: "prompted" or "unprompted".

    Returns:
        Equal-length input and target LongTensors, shifted by one, with START prepended to the input
        and STOP appended to the target.

    Raises:
        ValueError: If variant is neither "prompted" nor "unprompted".
    """
    build = fim_prompted if normalize_mode(variant) == PROMPTED else fim_unprompted
    ids = tokenizer.encode(build(record.full_seq, record.idr_start, record.idr_end))
    input_ids = torch.tensor([tokenizer.start_id, *ids], dtype=torch.long)
    target_ids = torch.tensor([*ids, tokenizer.stop_id], dtype=torch.long)
    return input_ids, target_ids


class RecordDataset(Dataset):
    """Map-style dataset over Records, yielding (input_ids, target_ids, loss_mask) triples.

    Backed by either an in-memory sequence of Records or a memory-mapped RecordStore.
    """

    def __init__(
        self,
        records,
        tokenizer: Tokenizer | None = None,
        *,
        max_len: int = 1024,
        prompted_prob: float = 0.5,
        completion_only: bool = False,
        seed: int = 0,
    ) -> None:
        """Build the dataset, dropping records too long to fit max_len.

        Args:
            records (Iterable[Record] | RecordStore): Records to serve, or a memory-mapped store.
            tokenizer: Tokenizer; defaults to Tokenizer().
            max_len: Maximum model positions; longer records are dropped.
            prompted_prob: Probability that a sample uses the prompted variant.
            completion_only: If True, mask the loss to the IDR completion; if False, to every token.
            seed: Seed for FIM variant selection.
        """
        self.tok = tokenizer or Tokenizer()
        self.max_len = int(max_len)
        self.prompted_prob = float(prompted_prob)
        self.completion_only = bool(completion_only)
        self._rng = random.Random(seed)

        keep = max_protein_len(self.max_len)  # filter on the full-context length so any sample fits
        if isinstance(records, RecordStore):
            self.store: RecordStore | None = records
            self.records = None
            self._keep = np.nonzero(records.seq_lengths() <= keep)[0]
            n_total = len(records)
            n_kept = len(self._keep)
        else:
            self.store = None
            records = list(records)
            self.records = [r for r in records if len(r.full_seq) <= keep]
            self._keep = None
            n_total = len(records)
            n_kept = len(self.records)
        if n_total - n_kept:
            log.info(
                f"RecordDataset: dropped {n_total - n_kept} record(s) longer than max_len={self.max_len}"
            )

    def __len__(self) -> int:
        """Return the number of records retained by the dataset length filter."""
        return len(self._keep) if self.store is not None else len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids, loss_mask) for kept record i.

        All tensors have equal length. The boolean mask selects loss-bearing targets;
        completion-only mode includes the IDR and STOP.
        """
        rec = self.store[int(self._keep[i])] if self.store is not None else self.records[i]
        variant = PROMPTED if self._rng.random() < self.prompted_prob else UNPROMPTED
        x, y = record_to_example(rec, self.tok, variant=variant)
        if self.completion_only:
            # The final targets are the IDR residues followed by STOP
            idr_len = rec.idr_end - rec.idr_start
            mask = torch.zeros(y.size(0), dtype=torch.bool)
            mask[-(idr_len + 1) :] = True
        else:
            mask = torch.ones(y.size(0), dtype=torch.bool)
        return x, y, mask


def make_collate(pad_id: int) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return a collator producing right-padded [B, L] (input, target, loss_mask) tensors.

    Pad input and target with pad_id and the boolean loss mask with False.
    """

    def collate(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad token inputs, targets, and loss masks to the longest sequence in the batch."""
        inputs, targets, masks = zip(*batch)
        x = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        y = pad_sequence(targets, batch_first=True, padding_value=pad_id)
        m = pad_sequence(masks, batch_first=True, padding_value=False)
        return x, y, m

    return collate
