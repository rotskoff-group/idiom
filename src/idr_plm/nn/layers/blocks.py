import torch.nn as nn
import torch.nn.functional as F
from .mha import MultiHeadAttention
from .ida import InteratomicDistanceAttention
from .geometric_attention import GeometricAttention


def swiglu_correction_fn(expansion_ratio, d_model):
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model, expansion_ratio, bias):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    """
    A unified transformer block that can optionally incorporate geometric attention.

    This class defines a transformer block that can be configured to use geometric attention
    alongside the standard multi-head attention mechanism. It is designed to be a flexible
    component of transformer-based models, allowing for the integration of geometric reasoning.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    use_geom_attn : bool, optional
        Whether to use geometric attention in addition to the standard multi-head attention. Defaults to False.
    v_heads : int, optional
        The number of heads to use for the geometric attention mechanism, if enabled. Must be specified if `use_geom_attn` is True.
    """

    def __init__(
        self,
        d_model,
        bias=False,
        use_mha=False,
        use_ga=False,
        use_ida=False,
        ffn_type="swiglu",
        scaling_factor=1,
        expansion_ratio=8 / 3,
        mha_args=None,
        gha_args=None,
        ida_args=None,
    ):
        super().__init__()
        self.use_mha = use_mha
        if self.use_mha:
            self.attn = MultiHeadAttention(d_model=d_model, **mha_args)
        self.use_ga = use_ga
        if self.use_ga:
            self.geom_attn = GeometricAttention(d_model=d_model, **gha_args)
        self.use_ida = use_ida
        if self.use_ida:
            self.interatomic_distance_attention = InteratomicDistanceAttention(
                d_model=d_model, **ida_args
            )
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = scaling_factor

    def forward(
        self,
        x,
        sequence_id,
        affine=None,
        affine_mask=None,
        coords=None,
        coords_mask=None,
    ):
        """
        Forward pass for the UnifiedTransformerBlock.

        Parameters
        ----------
        x : torch.Tensor[float]
            Input tensor to the transformer block, typically the output from the previous layer.
        sequence_id : torch.Tensor[int]
            Tensor containing sequence IDs for each element in the batch, used for attention masking.
        affine : Affine3D
            Affine3D class containing the translational and rotational components of the transformation
        affine_mask : torch.Tensor[bool]
            Boolean mask tensor indicating valid frames for geometric attention.

        Returns
        -------
        torch.Tensor[float]
            The output tensor after applying the transformer block operations.
        """
        if self.use_mha:
            r1 = self.attn(x=x, sequence_id=sequence_id)
            x = x + r1 / self.scaling_factor

        if self.use_ga:
            r2 = self.geom_attn(
                x, affine=affine, affine_mask=affine_mask, sequence_id=sequence_id
            )
            x = x + r2 / self.scaling_factor

        if self.use_ida:
            r3 = self.interatomic_distance_attention(
                x, coords=coords, coords_mask=coords_mask
            )
            x = x + r3 / self.scaling_factor

        r4 = self.ffn(x) / self.scaling_factor
        x = x + r4

        return x
