"""Streaming activation store for SAE training (D3) — no activation h5.

Runs the frozen model over record token-sequences, pulls residue-only residual-stream
activations at one layer (via :func:`idiom.model.activations.extract_activations`), and serves
them as shuffled ``[sae_batch_size, d_model]`` batches. A shuffling buffer decorrelates rows
within and across sequences. Activations are regenerated each epoch (D3): no caching to disk.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

from idiom.data.tokenizer import Tokenizer
from idiom.model.activations import extract_activations


class ActivationStore(IterableDataset):
    def __init__(
        self,
        model,
        record_loader,
        layer: int,
        *,
        sae_batch_size: int = 4096,
        buffer_size: int = 262_144,
        device: str | torch.device = "cpu",
        tokenizer: Tokenizer | None = None,
        drop_markers: bool = True,
    ) -> None:
        self.model = model.eval().to(device)
        self.record_loader = record_loader  # yields (input, target, mask) or input tokens [B, L]
        self.layer = layer
        self.sae_batch_size = sae_batch_size
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.tok = tokenizer or Tokenizer()
        self.drop_markers = drop_markers

    def _input_tokens(self, batch) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch  # RecordDataset yields a triple
        return x.to(self.device)

    @torch.no_grad()
    def _acts(self, tokens: torch.Tensor) -> torch.Tensor:
        out = extract_activations(
            self.model, tokens, [self.layer], tokenizer=self.tok, drop_markers=self.drop_markers
        )
        return out[self.layer].values  # [N_residues, d_model]

    @torch.no_grad()
    def __iter__(self):
        buf: list[torch.Tensor] = []
        n = 0
        for batch in self.record_loader:
            buf.append(self._acts(self._input_tokens(batch)))
            n += buf[-1].size(0)
            if n >= self.buffer_size:
                yield from self._drain(buf)
                n = sum(t.size(0) for t in buf)  # _drain leaves the (< batch) remainder
        # final flush of whatever full batches remain
        yield from self._drain(buf, final=True)

    def _drain(self, buf: list[torch.Tensor], *, final: bool = False):
        pool = torch.cat(buf, dim=0)
        pool = pool[torch.randperm(pool.size(0), device=pool.device)]  # shuffle the buffer
        full = (pool.size(0) // self.sae_batch_size) * self.sae_batch_size
        for i in range(0, full, self.sae_batch_size):
            yield pool[i : i + self.sae_batch_size]
        buf.clear()
        remainder = pool[full:]
        if not final and remainder.size(0):
            buf.append(remainder)  # carry the tail to the next buffer fill

    @torch.no_grad()
    def mean_activation(self, max_batches: int = 4) -> torch.Tensor:
        """Mean activation over a few record batches — seeds the SAE decoder bias (sparsify)."""
        total, count = None, 0
        for i, batch in enumerate(self.record_loader):
            acts = self._acts(self._input_tokens(batch))
            total = acts.sum(0) if total is None else total + acts.sum(0)
            count += acts.size(0)
            if i + 1 >= max_batches:
                break
        return (total / count).cpu()
