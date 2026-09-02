"""Reconstruction and sparsity of an SAE over held-out activations.

Measures the autoencoder in isolation, without the host model's loss:

- reconstruction: the fraction of variance unexplained and the explained variance, 1 - SSE/SST;
- sparsity: mean L0, the number of active latents per token, and the fraction of latents that
  never fire;
- feature density: the per-latent firing frequency.

Activations are drawn through an ActivationStore, using the region and prompt format the SAE was
trained on.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.io import read_records


@dataclass
class ReconstructionStats:
    """Reconstruction and sparsity summary of an SAE over held-out activations.

    Attributes:
        fvu (float): Fraction of variance unexplained.
        explained_var (float): 1 - fvu.
        l0_mean (float): Mean number of active latents per activation row.
        n_active_rows (int): Number of activation rows measured.
        n_dead (int): Number of latents that never fired.
        num_latents (int): Total number of latents.
        frac_dead (float): n_dead divided by num_latents.
        feature_freq (np.ndarray): Per-latent firing frequency, shape [num_latents].
    """

    fvu: float
    explained_var: float
    l0_mean: float
    n_active_rows: int
    n_dead: int
    num_latents: int
    frac_dead: float
    feature_freq: np.ndarray


@torch.no_grad()
def reconstruction_stats(
    host,
    sae,
    layer: int,
    region: str,
    prompted_prob: float,
    fasta: str,
    *,
    device,
    max_records: int | None = None,
    record_batch_size: int = 16,
    sae_batch_size: int = 4096,
) -> ReconstructionStats:
    """Compute reconstruction and sparsity statistics over a held-out record FASTA.

    Args:
        host: An idiom.IDiom, providing .model and .tok.
        sae: An idiom.IDiomSAE, whose .sae is the SparseCoder to measure.
        layer (int): The residual-stream layer the SAE was trained on.
        region (str): Residues to measure: "all", "idr", or "non_idr".
        prompted_prob (float): Probability of the prompted variant; 0.0 or 1.0 to match the SAE's
            recorded fim_mode.
        fasta (str): Held-out record FASTA.
        device: Device to run extraction and encoding on.
        max_records (int | None): Cap on records read from fasta, or None for all of them.
        record_batch_size (int): Records per forward batch.
        sae_batch_size (int): Activation rows per SAE batch.

    Returns:
        ReconstructionStats: The reconstruction and sparsity summary.

    Raises:
        ValueError: If no activations are extracted.
    """
    # Deferred: idiom.sae.training.__init__ pulls in LitSAE and therefore Lightning, which would
    # otherwise land on the plain `import idiom` inference path (~0.8 s and a training-only dep).
    from idiom.sae.training.activation_store import ActivationStore

    recs = read_records(fasta)
    if max_records:
        recs = itertools.islice(recs, max_records)
    ds = RecordDataset(list(recs), host.tok, max_len=host.model.cfg.max_seq_len,
                       prompted_prob=prompted_prob, completion_only=False)
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
