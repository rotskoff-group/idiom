from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional
from einops import rearrange
import logging
import numpy as np

log = logging.getLogger(__name__)


class InteratomicDistanceAttention(nn.Module):
    """Interatomic Distance Attention (IDA) layer.
    This layer computes attention based on the distances between atoms in a sequence.
    NB: IDA is intentionally *not* rotationally invariant / equivariant.
    Should be used with rotational data augmentation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        bias: bool = False,
        mask_mode: str = "none",
        dropout: bool = False,
        dropout_rate: float = 0.9,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.bias = bias
        self.mask_mode = mask_mode
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        spatial_dim = 3  # hardcoding 3D coordinates for atoms

        # Layer normalization for the input
        self.layer_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, spatial_dim * num_heads * 3, bias=bias)
        self.per_head_scalar = nn.Parameter(torch.ones(num_heads))
        self.out_proj = nn.Linear(spatial_dim * num_heads, d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        coords_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the IDA layer.

        Args:
            x: Input tensor of shape (B, S, C) where B is batch size, S is sequence length, and C is feature dimension.
            coords: Optional coordinate information for reasoning. Shape should be (B, S, 3) for 3D coordinates.
            coords_mask: Optional mask coordinates. Shape should be at least (S, S), and broadcastable to (B, H, S, S) where H is the number of heads.

        Returns:
            Output tensor after applying IDA.
        """
        if coords is None:
            coords = torch.full(
                (x.size(0), x.size(1), 3), float("-inf"), device=x.device
            )
        if self.dropout:
            if np.random.random() < self.dropout_rate:
                coords = torch.full_like(coords, float("-inf"))
        invalid_mask = torch.all(torch.isfinite(coords), dim=-1)
        coords[~invalid_mask] = 0.0
        if self.mask_mode == "causal":
            # create the coords_mask to mask the upper triangular part of the attention matrix
            coords_mask = torch.triu(
                torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1
            ).bool()

        # Q_d, K_d, V matrices
        q, k, v = self.proj(self.layer_norm(x)).chunk(3, dim=-1)
        reshape_str = "b s (h d) -> b h s d"
        q = rearrange(q, reshape_str, h=self.num_heads)
        k = rearrange(k, reshape_str, h=self.num_heads)
        v = rearrange(v, reshape_str, h=self.num_heads)

        # shift q and k by coords and then compute distances
        shift = coords.unsqueeze(1)
        q = q + shift
        k = k + shift
        # Compute pairwise distances
        d = 1.0 / sqrt(3) * torch.cdist(q, k, p=2)  # (B, H, S, S)
        a = F.softplus(self.per_head_scalar).view(1, self.num_heads, 1, 1) * d

        # Apply coordinate masking
        if coords_mask is not None:
            a = a.masked_fill(coords_mask, float("-inf"))

        a = nn.Softmax(dim=-1)(a)  # b h s s
        # multply a_{ij} with v_{jk}
        o = torch.einsum("bhij, bhjd -> bhid", a, v)  # b h s d
        o = rearrange(o, "b h s d -> b s (h d)")
        return x + self.out_proj(o)
