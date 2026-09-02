"""Prompt source for GRPO, assembled on the fly (no separate RL dataset file).

Prompts are FIM generation prefixes (1{prefix}3{suffix}2): the bare "132" prompt for unprompted
(de novo) optimization via unprompted_prompts, or one protein's flanks for prompted optimization
via record_prompts. A batch is assumed equal-length (the typical case: a single prompt repeated, or
one compartment's flank prompt).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from idiom.data.fim import fim_prompt
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer


class PromptDataset(Dataset):
    """Map-style dataset of encoded GRPO generation prompts."""

    def __init__(self, prompts: list[str], tokenizer: Tokenizer | None = None) -> None:
        self.tok = tokenizer or Tokenizer()
        self.encoded = [torch.tensor(self.tok.encode(p), dtype=torch.long) for p in prompts]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.encoded[i]


def unprompted_prompts(n: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """Return n copies of the bare "132" prompt for unprompted (de novo) optimization.

    Every prompt is identical, so a batch is trivially equal-length and each one expands into its
    own GRPO group of completions.

    Args:
        n (int): Number of prompt copies to produce.
        tokenizer (Tokenizer | None): Character tokenizer (a default is used if None).

    Returns:
        PromptDataset: Dataset of n identical de novo prompts.
    """
    return PromptDataset([fim_prompt()] * n, tokenizer)


def record_prompts(fasta: str, n_per: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """Return n_per copies of each record's flank prompt for prompted optimization.

    Args:
        fasta (str): Path to the FASTA of records to draw flank prompts from.
        n_per (int): Number of copies per record.
        tokenizer (Tokenizer | None): Character tokenizer (a default is used if None).

    Returns:
        PromptDataset: Dataset of flank prompts, n_per per record.
    """
    prompts = [fim_prompt(r.full_seq, r.idr_start, r.idr_end) for r in read_records(fasta)]
    return PromptDataset([p for p in prompts for _ in range(n_per)], tokenizer)


def collate_prompts(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack equal-length prompts into a [B, P] batch.

    Length-bucket upstream if prompt lengths differ.

    Args:
        batch (list[torch.Tensor]): Equal-length encoded prompts.

    Returns:
        torch.Tensor: Stacked prompts of shape [B, P].
    """
    return torch.stack(batch)
