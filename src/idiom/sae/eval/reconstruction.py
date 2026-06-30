"""Reconstruction + sparsity of an SAE over held-out activations.

These metrics describe the SAE in isolation (no downstream model loss): how well it reconstructs
the residual stream and how it spends its latents. They contextualise the downstream "loss
recovered" of :mod:`idiom.sae.eval.fidelity` and feed the SAE figures.

  - **reconstruction** — FVU (fraction of variance unexplained) and explained variance over the
    held-out activations, ``1 - SSE/SST``.
  - **sparsity** — mean L0 (active latents per token; ~k for a top-k SAE) and the dead-feature
    fraction (latents that never fire on the held-out set).
  - **feature density** — per-latent activation frequency, for the feature-density histogram.

Activations are pulled with the SAE's training ``region`` (residue-masking is applied by the
:class:`~idiom.sae.training.activation_store.ActivationStore`, dropping START / FIM-marker /
control positions) and prompt format (``fim_idr_prob``), so the SAE is measured on-distribution.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReconstructionStats:
    fvu: float
    explained_var: float
    l0_mean: float
    n_active_rows: int
    n_dead: int
    num_latents: int
    frac_dead: float
    feature_freq: np.ndarray  # per-latent firing frequency, shape (num_latents,)


@torch.no_grad()
def reconstruction_stats(
    host,
    sae,
    layer: int,
    region: str,
    fim_idr_prob: float,
    fasta: str,
    *,
    device,
    max_records: int | None = None,
    record_batch_size: int = 16,
    sae_batch_size: int = 4096,
) -> ReconstructionStats:
    """FVU / explained variance / mean-L0 / dead fraction / per-feature firing freq over ``fasta``.

    Args:
        host: an :class:`~idiom.IDiom` (provides ``.model`` and ``.tok``).
        sae: an :class:`~idiom.IDiomSAE`; its ``.sae`` is the :class:`SparseCoder`.
        layer: residual-stream layer the SAE was trained on.
        region: the SAE's training region (``"all"`` | ``"idr"`` | ``"non_idr"``).
        fim_idr_prob: prompt format to match the SAE's ``fim_mode`` (0.0 idp / 1.0 idr).
        fasta: held-out record FASTA.
    """
    from torch.utils.data import DataLoader

    from idiom.data.dataset import RecordDataset, make_collate
    from idiom.data.io import read_records
    from idiom.sae.training.activation_store import ActivationStore

    recs = read_records(fasta)
    if max_records:
        recs = itertools.islice(recs, max_records)
    ds = RecordDataset(list(recs), host.tok, max_len=host.model.cfg.max_seq_len,
                       fim_idr_prob=fim_idr_prob, completion_only=False)
    dl = DataLoader(ds, batch_size=record_batch_size, collate_fn=make_collate(host.tok.pad_id))
    store = ActivationStore(host.model, dl, layer, sae_batch_size=sae_batch_size,
                            buffer_size=sae_batch_size * 8, device=device, tokenizer=host.tok,
                            region=region)
    sc = sae.sae.to(device).eval()
    n_lat = sc.W_dec.shape[0]
    sse = 0.0
    x_sum = torch.zeros(sc.b_dec.shape[0], device=device)
    x_sqsum = 0.0
    fire = torch.zeros(n_lat, device=device)
    l0_sum, n = 0.0, 0
    for x in store:
        x = x.to(device).float()
        dense = sc.encode_dense(x)
        x_hat = sc.decode_dense(dense)
        sse += float(((x - x_hat) ** 2).sum())
        x_sum += x.sum(0)
        x_sqsum += float((x ** 2).sum())
        act = dense > 0
        fire += act.sum(0)
        l0_sum += float(act.sum())
        n += x.shape[0]
    if n == 0:
        raise ValueError("no activations extracted")
    total_sse = x_sqsum - float((x_sum ** 2).sum()) / n  # sum ||x - mean||^2
    fvu = sse / total_sse if total_sse > 0 else float("nan")
    return ReconstructionStats(
        fvu=fvu,
        explained_var=1.0 - fvu,
        l0_mean=l0_sum / n,
        n_active_rows=n,
        n_dead=int((fire == 0).sum()),
        num_latents=n_lat,
        frac_dead=float((fire == 0).float().mean()),
        feature_freq=(fire / n).cpu().numpy(),
    )
