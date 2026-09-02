"""Root-mean-square layer normalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with a learned scale and no bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Build the norm with a unit-initialized scale.

        Args:
            dim (int): Size of the normalized last dimension.
            eps (float): Value added to the mean square before the reciprocal square root.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x by its root mean square over the last dimension and apply the scale.

        Args:
            x (Tensor): Input of shape [..., dim].

        Returns:
            Tensor: The normalized, scaled tensor, in the input dtype.
        """
        # Normalize by RMS over the last dim, in float32 for stability, then rescale.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight
