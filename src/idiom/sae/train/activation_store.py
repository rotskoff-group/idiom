"""Streaming activation store for SAE training.

Runs a frozen model over batches of record token sequences, extracts residual-stream activations
at one layer, and serves them as shuffled [sae_batch_size, d_model] batches. Rows accumulate in a
buffer that is shuffled and drained once it reaches buffer_size; nothing is written to disk.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset

from idiom.data.tokenizer import Tokenizer
from idiom.model.activations import extract_activations


class ActivationStore(IterableDataset):
    """Iterable dataset yielding shuffled residual-stream activation batches.

    Attributes:
        model: The frozen host transformer, in eval mode on the store's device.
        record_loader: The loader supplying token sequences.
        layer (int): The residual-stream layer being extracted.
        sae_batch_size (int): Rows per yielded batch.
        buffer_size (int): Rows accumulated before the buffer is shuffled and drained.
        device (torch.device): Device extraction and shuffling run on.
        region (str): Residues kept: "all", "idr", or "non_idr".
    """

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
        region: str = "all",
    ) -> None:
        """Configure the store and move the model to the target device in eval mode.

        Args:
            model: The host transformer to extract activations from.
            record_loader: A loader yielding either (input, target, mask) triples or input token
                tensors of shape [B, L].
            layer (int): The residual-stream layer to extract.
            sae_batch_size (int): Number of activation rows per yielded batch.
            buffer_size (int): Number of rows to accumulate before shuffling and draining.
            device (str | torch.device): Device to run extraction and shuffling on.
            tokenizer (Tokenizer | None): Tokenizer for region masking; a default if None.
            drop_markers (bool): If True, keep only real residues; if False, also keep FIM markers.
            region (str): Residues to keep: "all", "idr", or "non_idr".
        """
        self.model = model.eval().to(device)
        self.record_loader = record_loader  # yields (input, target, mask) or input tokens [B, L]
        self.layer = layer
        self.sae_batch_size = sae_batch_size
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.tok = tokenizer or Tokenizer()
        self.drop_markers = drop_markers
        self.region = region  # "all" | "idr" | "non_idr" (residues kept relative to the '2' marker)

    def _input_tokens(self, batch) -> torch.Tensor:
        """Return the input token tensor from a batch, which may be a triple or a bare tensor."""
        x = batch[0] if isinstance(batch, (tuple, list)) else batch  # RecordDataset yields a triple
        return x.to(self.device)

    @torch.no_grad()
    def _acts(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the [N_residues, d_model] activations for one batch of token sequences."""
        out = extract_activations(
            self.model, tokens, [self.layer], tokenizer=self.tok,
            drop_markers=self.drop_markers, region=self.region,
        )
        return out[self.layer].values  # [N_residues, d_model]

    @torch.no_grad()
    def __iter__(self):
        """Yield shuffled activation batches until the record loader is exhausted.

        Yields:
            torch.Tensor: A batch of shape [sae_batch_size, d_model]. Rows left over after the
                final drain are discarded.
        """
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
        """Shuffle the buffer and yield full batches, carrying any remainder unless final."""
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
        """Return the mean activation over the first few record batches.

        Args:
            max_batches (int): Number of record batches to average over.

        Returns:
            torch.Tensor: The mean activation vector of shape [d_model], on CPU.
        """
        total, count = None, 0
        for i, batch in enumerate(self.record_loader):
            acts = self._acts(self._input_tokens(batch))
            total = acts.sum(0) if total is None else total + acts.sum(0)
            count += acts.size(0)
            if i + 1 >= max_batches:
                break
        return (total / count).cpu()
