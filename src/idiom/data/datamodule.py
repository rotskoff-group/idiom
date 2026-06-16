"""Lightning DataModule over per-split record FASTAs.

The curation step writes ``train.fasta`` / ``val.fasta`` / ``test.fasta`` in the shared
``_IDR_x-y`` record format (D16); the split is decided there, so this module just reads each
file into a :class:`~idiom.data.dataset.RecordDataset` and serves batches. Tokenization and
FIM assembly happen on the fly inside the dataset (no precompute).
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer


class RecordDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_fasta: str | Path,
        val_fasta: str | Path | None = None,
        test_fasta: str | Path | None = None,
        *,
        tokenizer: Tokenizer | None = None,
        max_len: int = 1024,
        fim_full_prob: float = 0.5,
        completion_only: bool = False,  # True for SFT (loss on the IDR completion only)
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.train_fasta = train_fasta
        self.val_fasta = val_fasta
        self.test_fasta = test_fasta
        self.tok = tokenizer or Tokenizer()
        self.max_len = max_len
        self.fim_full_prob = fim_full_prob
        self.completion_only = completion_only
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_set: RecordDataset | None = None
        self.val_set: RecordDataset | None = None
        self.test_set: RecordDataset | None = None

    def _build(self, path: str | Path) -> RecordDataset:
        return RecordDataset(
            read_records(path),
            self.tok,
            max_len=self.max_len,
            fim_full_prob=self.fim_full_prob,
            completion_only=self.completion_only,
            seed=self.seed,
        )

    def setup(self, stage: str | None = None) -> None:
        # Idempotent: Lightning may call setup() more than once; only build each split once.
        if self.train_set is None:
            self.train_set = self._build(self.train_fasta)
        if self.val_fasta is not None and self.val_set is None:
            self.val_set = self._build(self.val_fasta)
        if self.test_fasta is not None and self.test_set is None:
            self.test_set = self._build(self.test_fasta)

    def _loader(self, dataset: RecordDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=make_collate(self.tok.pad_id),
            drop_last=shuffle,  # drop the ragged tail only during training
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        # None -> Lightning skips validation entirely (e.g. SFT with no held-out set)
        return self._loader(self.val_set, shuffle=False) if self.val_set is not None else None

    def test_dataloader(self) -> DataLoader | None:
        return self._loader(self.test_set, shuffle=False) if self.test_set is not None else None
