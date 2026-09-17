"""Top-k and group-max sparse autoencoders with FVU, AuxK, and Multi-TopK losses."""

from __future__ import annotations

from typing import Literal, NamedTuple

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class EncoderOutput(NamedTuple):
    """Selected latents and pre-selection activations.

    Attributes:
        top_acts: Activations of the selected latents, shape [..., k].
        top_indices: Indices of the selected latents, shape [..., k].
        pre_acts: Post-ReLU activations before selection, shape [..., num_latents].
    """

    top_acts: Tensor
    top_indices: Tensor
    pre_acts: Tensor


class ForwardOutput(NamedTuple):
    """SAE reconstruction, selected latents, and losses.

    Attributes:
        sae_out: The reconstruction, shape [..., d_in].
        latent_acts: Activations of the selected latents, shape [..., k].
        latent_indices: Indices of the selected latents, shape [..., k].
        fvu: Scalar fraction of variance unexplained.
        auxk_loss: Scalar dead-latent revival loss; 0 when no latents are marked dead.
        multi_topk_fvu: Scalar Multi-TopK FVU; 0 unless multi_topk is set.
    """

    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor
    fvu: Tensor
    auxk_loss: Tensor
    multi_topk_fvu: Tensor


class SparseCoder(nn.Module):
    """Top-k or group-max sparse autoencoder over a layer's residual stream.

    Attributes:
        k: Scalar tensor storing the number of selected latents per token.
        W_dec: Decoder weights, shape [num_latents, d_in].
        b_dec: Decoder bias, shape [d_in].
    """

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
        """Initialize the decoder from encoder weights, optionally normalizing its rows.

        Args:
            d_in: Input (residual-stream) dimension.
            num_latents: Number of latents; 0 uses d_in * expansion_factor.
            expansion_factor: Latents-per-input multiplier used when num_latents is 0.
            k: Number of latents kept active per token.
            activation: Selection rule: global top-k, or the maximum within each of k equal groups
                of latents.
            multi_topk: If True, also compute the Multi-TopK auxiliary FVU.
            normalize_decoder: If True, initialize decoder rows to unit norm.
            device: Device for the parameters.
            dtype: Dtype for the parameters.

        Raises:
            ValueError: If activation is "groupmax" and num_latents is not divisible by k.
        """
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents or d_in * expansion_factor
        self.activation = activation
        self.multi_topk = multi_topk
        self.normalize_decoder = normalize_decoder

        if activation == "groupmax" and self.num_latents % k != 0:
            raise ValueError(f"groupmax requires num_latents ({self.num_latents}) divisible by k ({k})")

        # k stored as a buffer so checkpoints are self-contained
        self.register_buffer("k", torch.tensor(int(k), dtype=torch.long))

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))

    @property
    def device(self) -> torch.device:
        """Return the device of the encoder weights."""
        return self.encoder.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the encoder weights."""
        return self.encoder.weight.dtype

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode an input by subtracting b_dec, projecting, applying ReLU, and selecting latents.

        Args:
            x: Input activations of shape [..., d_in].

        Returns:
            An EncoderOutput containing selected activations and indices, unsorted,
            and post-ReLU activations before selection.
        """
        pre_acts = F.relu(self.encoder(x - self.b_dec))
        k = int(self.k)

        if self.activation == "groupmax":
            values, indices = pre_acts.unflatten(-1, (k, -1)).max(dim=-1)
            # convert per-group indices into flat latent indices
            offsets = torch.arange(0, self.num_latents, self.num_latents // k, device=pre_acts.device)
            indices = offsets + indices
        else:
            values, indices = pre_acts.topk(k, dim=-1, sorted=False)

        return EncoderOutput(values, indices, pre_acts)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode a sparse latent set as a weighted sum of decoder rows, plus b_dec.

        Args:
            top_acts: Activations of the selected latents, shape [..., k].
            top_indices: Latent indices of the selected latents, shape [..., k].

        Returns:
            The reconstruction of shape [..., d_in].
        """
        chosen = self.W_dec[top_indices]  # [..., k, d_in]
        return (top_acts.unsqueeze(-1) * chosen).sum(dim=-2) + self.b_dec

    def encode_dense(self, x: Tensor) -> Tensor:
        """Return [..., num_latents] activations for x [..., d_in], zero outside selected latents."""
        top_acts, top_indices, _ = self.encode(x)
        out = x.new_zeros(*x.shape[:-1], self.num_latents)
        return out.scatter_(-1, top_indices, top_acts.to(out.dtype))

    def decode_dense(self, f: Tensor) -> Tensor:
        """Decode [..., num_latents] activations to [..., d_in], including b_dec."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: Tensor, *, dead_mask: Tensor | None = None) -> ForwardOutput:
        """Encode, decode, and compute the reconstruction and auxiliary losses.

        All three losses are normalized by the total variance of x within the batch.

        Args:
            x: Input activations of shape [..., d_in].
            dead_mask: Boolean mask over latents marking dead ones. When given and any are dead, the
                AuxK loss fits those latents to the reconstruction residual.

        Returns:
            A ForwardOutput containing the reconstruction, selected activations and indices,
            and scalar FVU, AuxK, and Multi-TopK losses. Auxiliary losses are 0 when unused.
        """
        top_acts, top_indices, pre_acts = self.encode(x)
        sae_out = self.decode(top_acts, top_indices)

        e = x - sae_out
        total_variance = (x - x.mean(0)).pow(2).sum()

        # AuxK: encourage the top ~half of dead latents to predict the residual
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

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Rescale every decoder row to unit norm, in place."""
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = self.W_dec.data.norm(dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """Subtract from the decoder gradient its component parallel to each decoder row, in place.

        Raises:
            AssertionError: If the decoder has no gradient.
        """
        assert self.W_dec.grad is not None
        parallel = einops.einsum(self.W_dec.grad, self.W_dec.data, "f d, f d -> f")
        self.W_dec.grad -= einops.einsum(parallel, self.W_dec.data, "f, f d -> f d")
