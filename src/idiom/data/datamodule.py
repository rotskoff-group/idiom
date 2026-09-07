"""Lightning dataloaders for per-split record FASTAs."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.record_store import open_or_build
from idiom.data.tokenizer import Tokenizer


class RecordDataModule(L.LightningDataModule):
    """Lightning dataloaders backed by per-split record stores.

    Attributes:
        train_set (RecordDataset | None): The train split, built by setup().
        val_set (RecordDataset | None): The validation split, or None if no val_fasta was given.
        test_set (RecordDataset | None): The test split, or None if no test_fasta was given.
    """

    def __init__(
        self,
        train_fasta: str | Path,
        val_fasta: str | Path | None = None,
        test_fasta: str | Path | None = None,
        *,
        tokenizer: Tokenizer | None = None,
        max_len: int = 1024,
        prompted_prob: float = 0.5,
        completion_only: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        """Configure splits for lazy loading in setup().

        Args:
            train_fasta: Record FASTA for the train split.
            val_fasta: Record FASTA for validation, or None to skip validation.
            test_fasta: Record FASTA for test, or None to skip testing.
            tokenizer: Tokenizer; defaults to Tokenizer().
            max_len: Maximum model positions; longer records are dropped.
            prompted_prob: Probability that a sample uses the prompted variant.
            completion_only: If True, mask the loss to the IDR completion.
            batch_size: Batch size for all dataloaders.
            num_workers: DataLoader worker processes.
            seed: Seed for FIM variant selection.
        """
        super().__init__()
        self.train_fasta = train_fasta
        self.val_fasta = val_fasta
        self.test_fasta = test_fasta
        self.tok = tokenizer or Tokenizer()
        self.max_len = max_len
        self.prompted_prob = prompted_prob
        self.completion_only = completion_only
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_set: RecordDataset | None = None
        self.val_set: RecordDataset | None = None
        self.test_set: RecordDataset | None = None

    def _build(self, path: str | Path) -> RecordDataset:
        # Memory mapping lets DDP ranks share record data through the OS page cache
        return RecordDataset(
            open_or_build(path),
            self.tok,
            max_len=self.max_len,
            prompted_prob=self.prompted_prob,
            completion_only=self.completion_only,
            seed=self.seed,
        )

    def setup(self, stage: str | None = None) -> None:
        """Build configured splits once; stage is ignored."""
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
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """Return a shuffled DataLoader over the train split, dropping the ragged last batch."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        """Return a DataLoader over the validation split, or None if there is no val_fasta."""
        return self._loader(self.val_set, shuffle=False) if self.val_set is not None else None

    def test_dataloader(self) -> DataLoader | None:
        """Return a DataLoader over the test split, or None if there is no test_fasta."""
        return self._loader(self.test_set, shuffle=False) if self.test_set is not None else None
