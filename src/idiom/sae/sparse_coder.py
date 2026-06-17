"""Top-k / group-max sparse autoencoder for IDiom activations.

Faithful to EleutherAI ``sparsify`` (``sparsify/sparse_coder.py``,
``fused_encoder.py``) and OpenAI ``sparse_autoencoder``:

- encoder applies ``relu`` then top-k selection (sparsify ``fused_encoder``);
- ``b_dec`` is subtracted before the encoder and added back after the decoder;
- the decoder is a single weight ``W_dec`` of shape ``[num_latents, d_in]`` with
  unit-norm rows (one unit vector per latent), re-normalized every step;
- the gradient component parallel to each decoder row is projected out before the step;
- the main loss is the **fraction of variance unexplained (FVU)**, normalized by the
  batch variance, exactly as in sparsify (not raw summed MSE);
- the AuxK loss revives dead latents by having the top ``d_in//2`` dead latents predict the
  reconstruction residual, normalized by total variance and down-weighted when few are dead;
- optional **Multi-TopK** auxiliary loss (Gao et al. 2024, "progressive recovery").

``forward`` returns a :class:`ForwardOutput` with the losses pre-computed (sparsify style);
the LightningModule just combines them.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""
    top_indices: Tensor
    """Indices of the top-k latents."""
    pre_acts: Tensor
    """Post-ReLU activations before top-k selection (used for the AuxK loss)."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor
    fvu: Tensor
    """Fraction of variance unexplained (the main loss)."""
    auxk_loss: Tensor
    """Dead-latent revival loss (0 if no dead mask / no dead latents)."""
    multi_topk_fvu: Tensor
    """Multi-TopK FVU (0 unless ``multi_topk``)."""


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        *,
        num_latents: int = 0,
        expansion_factor: int = 8,
        k: int = 32,
        activation: Literal["topk", "groupmax"] = "topk",
        multi_topk: bool = False,
        normalize_decoder: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents or d_in * expansion_factor
        self.activation = activation
        self.multi_topk = multi_topk
        self.normalize_decoder = normalize_decoder

        if activation == "groupmax" and self.num_latents % k != 0:
            raise ValueError(
                f"groupmax requires num_latents ({self.num_latents}) divisible by k ({k})"
            )

        # k stored as a buffer so checkpoints are self-contained.
        self.register_buffer("k", torch.tensor(int(k), dtype=torch.long))

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        # Decoder: one unit-norm row per latent. Initialized to the encoder weights
        # (sparsify's "tied" init), then row-normalized.
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())  # [num_latents, d_in]
        if normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))

    @property
    def device(self) -> torch.device:
        return self.encoder.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.weight.dtype

    # --- encode / decode ---
    def encode(self, x: Tensor) -> EncoderOutput:
        """ReLU then top-k (sparsify ``fused_encoder``). ``pre_acts`` is post-ReLU."""
        pre_acts = F.relu(self.encoder(x - self.b_dec))
        k = int(self.k)

        if self.activation == "groupmax":
            values, indices = pre_acts.unflatten(-1, (k, -1)).max(dim=-1)
            # convert per-group indices into flat latent indices
            offsets = torch.arange(
                0, self.num_latents, self.num_latents // k, device=pre_acts.device
            )
            indices = offsets + indices
        else:
            values, indices = pre_acts.topk(k, dim=-1, sorted=False)

        return EncoderOutput(values, indices, pre_acts)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Sparse decode: weighted sum of the selected unit-norm decoder rows + ``b_dec``."""
        chosen = self.W_dec[top_indices]  # [..., k, d_in]
        return (top_acts.unsqueeze(-1) * chosen).sum(dim=-2) + self.b_dec

    def encode_dense(self, x: Tensor) -> Tensor:
        """Full ``[..., num_latents]`` activation vector (top-k sparse, densified).

        Convenience for analysis/steering — not used in the training loss path.
        """
        top_acts, top_indices, _ = self.encode(x)
        out = x.new_zeros(*x.shape[:-1], self.num_latents)
        return out.scatter_(-1, top_indices, top_acts.to(out.dtype))

    def decode_dense(self, f: Tensor) -> Tensor:
        """Decode from a dense ``[..., num_latents]`` latent vector."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: Tensor, *, dead_mask: Tensor | None = None) -> ForwardOutput:
        top_acts, top_indices, pre_acts = self.encode(x)
        sae_out = self.decode(top_acts, top_indices)

        e = x - sae_out
        total_variance = (x - x.mean(0)).pow(2).sum()

        # AuxK: encourage the top ~half of dead latents to predict the residual.
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = x.shape[-1] // 2  # heuristic from Gao et al. Appendix B.1
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = scale * (e_hat - e.detach()).pow(2).sum() / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        fvu = e.pow(2).sum() / total_variance

        if self.multi_topk:
            mt_acts, mt_indices = pre_acts.topk(4 * int(self.k), sorted=False)
            mt_out = self.decode(mt_acts, mt_indices)
            multi_topk_fvu = (mt_out - x).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(sae_out, top_acts, top_indices, fvu, auxk_loss, multi_topk_fvu)

    # --- decoder constraints (sparsify methods) ---
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = self.W_dec.data.norm(dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec.grad is not None
        parallel = einops.einsum(self.W_dec.grad, self.W_dec.data, "f d, f d -> f")
        self.W_dec.grad -= einops.einsum(parallel, self.W_dec.data, "f, f d -> f d")
