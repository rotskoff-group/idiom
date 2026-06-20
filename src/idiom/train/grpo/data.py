"""Prompt source for GRPO (on-the-fly, no RL h5).

Prompts are FIM generation prefixes (``1{prefix}3{suffix}2``): de-novo ``"132"`` for IDP
optimization (``idp_prompts``), or one protein's flanks for prompted-IDR optimization. A batch is assumed
equal-length (the typical case — a single prompt repeated, or one compartment's prompt).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from idiom.data.fim import fim_prompt
from idiom.data.io import read_records
from idiom.data.tokenizer import Tokenizer


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str], tokenizer: Tokenizer | None = None) -> None:
        self.tok = tokenizer or Tokenizer()
        self.encoded = [torch.tensor(self.tok.encode(p), dtype=torch.long) for p in prompts]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.encoded[i]


def idp_prompts(n: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """``n`` copies of the de-novo prompt ``"132"`` (IDP optimization)."""
    return PromptDataset([fim_prompt()] * n, tokenizer)


def record_prompts(fasta: str, n_per: int, tokenizer: Tokenizer | None = None) -> PromptDataset:
    """``n_per`` copies of each record's flank prompt (prompted-IDR optimization)."""
    prompts = [fim_prompt(r.full_seq, r.idr_start, r.idr_end) for r in read_records(fasta)]
    return PromptDataset([p for p in prompts for _ in range(n_per)], tokenizer)


def collate_prompts(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack equal-length prompts into ``[B, P]`` (length-bucket upstream if prompts differ)."""
    return torch.stack(batch)
