"""FIM prompt datasets and collation for equal-length batches."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from idiom.data.fim import fim_prompt
from idiom.data.records import read_records
from idiom.data.tokenizer import Tokenizer


class PromptDataset(Dataset):
    """Map-style dataset of encoded generation prompts."""

    def __init__(self, prompts: list[str], tokenizer: Tokenizer | None = None) -> None:
        """Encode the prompts once, at construction.

        Args:
            prompts: FIM prompt strings.
            tokenizer: Tokenizer; defaults to Tokenizer().
        """
        self.tok = tokenizer or Tokenizer()
        self.encoded = [torch.tensor(self.tok.encode(p), dtype=torch.long) for p in prompts]

    def __len__(self) -> int:
        """Return the number of encoded prompts."""
        return len(self.encoded)

    def __getitem__(self, i: int) -> torch.Tensor:
        """Return the encoded prompt at the requested index."""
        return self.encoded[i]


def unprompted_prompts(n: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """Return n encoded copies of "132", using the default tokenizer if omitted."""
    return PromptDataset([fim_prompt()] * n, tokenizer)


def prompted_prompts(fasta: str, n_per: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """Return a dataset holding n_per copies of each record's flank prompt.

    Flank prompts differ in length between records, so a batch must not mix records.

    Args:
        fasta: Path to the record FASTA to draw flank prompts from.
        n_per: Number of copies per record.
        tokenizer: Tokenizer; defaults to Tokenizer().

    Returns:
        Dataset of flank prompts, n_per consecutive copies per record.
    """
    prompts = [fim_prompt(r.full_seq, r.idr_start, r.idr_end) for r in read_records(fasta)]
    return PromptDataset([p for p in prompts for _ in range(n_per)], tokenizer)


def collate_prompts(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack equal-length prompt tensors into a [B, P] batch.

    Raises:
        RuntimeError: If prompt lengths differ.
    """
    return torch.stack(batch)
