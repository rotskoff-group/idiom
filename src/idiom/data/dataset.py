"""Training dataset mapping records to next-token-prediction pairs.

Each item assembles a FIM string from a Record (prompted or unprompted, chosen at random per
sample), tokenizes it, and forms the equal-length pair

    input  = [START, t0, t1, ..., t_{n-1}]
    target = [t0,    t1, ..., t_{n-1}, STOP]

together with a boolean loss mask. Padding is added by make_collate.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from loguru import logger as log
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from idiom.data.fim import PROMPTED, UNPROMPTED, fim_prompted, fim_unprompted, normalize_mode
from idiom.data.io import Record
from idiom.data.record_store import RecordStore
from idiom.data.tokenizer import Tokenizer

# A full example is START + 1{prefix}3{suffix}2{IDR}: the 3 FIM markers + START over the
# residues. So model positions = len(full_seq) + 4. Records longer than that are dropped.
FIM_OVERHEAD = 4


def max_protein_len(max_len: int) -> int:
    """Return the largest full_seq length whose example fits in max_len model positions.

    Args:
        max_len (int): Maximum number of model positions.

    Returns:
        int: max_len minus the START token and the three FIM markers.
    """
    return max_len - FIM_OVERHEAD


def record_to_example(
    record: Record, tokenizer: Tokenizer, *, variant: str = PROMPTED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (input_ids, target_ids) next-token pair for one record.

    Args:
        record (Record): The record to encode.
        tokenizer (Tokenizer): Character tokenizer for the FIM string.
        variant (str): "prompted" or "unprompted".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Equal-length input and target LongTensors, shifted by
            one, with START prepended to the input and STOP appended to the target.

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
            tokenizer (Tokenizer | None): Character tokenizer; a default Tokenizer if None.
            max_len (int): Maximum model positions; longer records are dropped.
            prompted_prob (float): Probability that a sample uses the prompted variant.
            completion_only (bool): If True, mask the loss to the IDR completion; if False, to
                every token.
            seed (int): Seed for the per-sample prompted/unprompted choice.
        """
        self.tok = tokenizer or Tokenizer()
        self.max_len = int(max_len)
        self.prompted_prob = float(prompted_prob)
        # completion_only=True -> SFT: compute loss only on the IDR completion (after the 2
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
            log.info(f"RecordDataset: dropped {n_total - n_kept} record(s) longer than "
                     f"max_len={self.max_len}")

    def __len__(self) -> int:
        return len(self._keep) if self.store is not None else len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the (input_ids, target_ids, loss_mask) triple for one record.

        Args:
            i (int): Index into the kept records.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Equal-length input ids, target ids,
                and a boolean mask selecting the target positions that contribute to the loss.
        """
        rec = self.store[int(self._keep[i])] if self.store is not None else self.records[i]
        # per-sample augmentation: prompted (context) with prob prompted_prob, else unprompted
        variant = PROMPTED if self._rng.random() < self.prompted_prob else UNPROMPTED
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
    """Build a collate function that right-pads (input, target, loss_mask) triples into batches.

    Args:
        pad_id (int): Token id used to pad the input and target; the loss mask is padded with False.

    Returns:
        Callable: A function mapping a list of triples to right-padded [B, L] tensors.
    """

    def collate(batch):
        inputs, targets, masks = zip(*batch)
        x = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        y = pad_sequence(targets, batch_first=True, padding_value=pad_id)
        m = pad_sequence(masks, batch_first=True, padding_value=False)  # pad positions: no loss
        return x, y, m

    return collate
