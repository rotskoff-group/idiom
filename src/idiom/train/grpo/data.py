"""Prompt datasets for GRPO.

Prompts are FIM generation prefixes ("1{prefix}3{suffix}2"): unprompted_prompts repeats the bare
"132" prompt, and prompted_prompts draws one flank prompt per record in a FASTA. collate_prompts
stacks prompts without padding, so a batch must be equal-length.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from idiom.data.fim import fim_prompt
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer


class PromptDataset(Dataset):
    """Map-style dataset of encoded generation prompts."""

    def __init__(self, prompts: list[str], tokenizer: Tokenizer | None = None) -> None:
        """Encode the prompts once, at construction.

        Args:
            prompts (list[str]): FIM prompt strings.
            tokenizer (Tokenizer | None): Character tokenizer; a default Tokenizer if None.
        """
        self.tok = tokenizer or Tokenizer()
        self.encoded = [torch.tensor(self.tok.encode(p), dtype=torch.long) for p in prompts]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.encoded[i]


def unprompted_prompts(n: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """Return a dataset of n copies of the bare "132" prompt.

    Args:
        n (int): Number of prompt copies to produce.
        tokenizer (Tokenizer | None): Character tokenizer; a default Tokenizer if None.

    Returns:
        PromptDataset: Dataset of n identical prompts.
    """
    return PromptDataset([fim_prompt()] * n, tokenizer)


def prompted_prompts(fasta: str, n_per: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """Return a dataset holding n_per copies of each record's flank prompt.

    Flank prompts differ in length between records, so a batch must not mix records.

    Args:
        fasta (str): Path to the record FASTA to draw flank prompts from.
        n_per (int): Number of copies per record.
        tokenizer (Tokenizer | None): Character tokenizer; a default Tokenizer if None.

    Returns:
        PromptDataset: Dataset of flank prompts, n_per consecutive copies per record.
    """
    prompts = [fim_prompt(r.full_seq, r.idr_start, r.idr_end) for r in read_records(fasta)]
    return PromptDataset([p for p in prompts for _ in range(n_per)], tokenizer)


def collate_prompts(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack equal-length prompts into one batch.

    Args:
        batch (list[torch.Tensor]): Encoded prompts, which must all have the same length.

    Returns:
        torch.Tensor: Stacked prompts of shape [B, P].

    Raises:
        RuntimeError: If the prompts differ in length.
    """
    return torch.stack(batch)
