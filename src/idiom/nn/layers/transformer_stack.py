import torch
import math
import torch.nn as nn
from .blocks import UnifiedTransformerBlock


class TransformerStack(nn.Module):
    """
    A stack of transformer blocks used in the ESM-3 model. Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless positions in the input.
            Only applies in the geometric attention blocks, which is conditioned on the structure
    """

    def __init__(
        self,
        d_model,
        n_layers,
        mha_layer_indices,
        bias=False,
        mha_args=None,
        scaling_factor=1.0,
        ffn_type="swiglu",
        norm_type="layer_norm",
        expansion_ratio=8 / 3,
    ):
        super().__init__()
        if scaling_factor is None:
            scaling_factor = math.sqrt(n_layers / 36)
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model=d_model,
                    bias=bias,
                    use_mha=(layer_num in mha_layer_indices),
                    ffn_type=ffn_type,
                    scaling_factor=scaling_factor,
                    expansion_ratio=expansion_ratio,
                    mha_args=mha_args,
                )
                for layer_num in range(n_layers)
            ]
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(d_model, bias=False)
        elif norm_type == "identity":
            self.norm = nn.Identity()
        else:
            raise ValueError("Unknown norm_type passed")

    def forward(self, x, sequence_id, return_hidden_states=False):
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
            return_hidden_states (bool): If True, return the output of each block as a list.

        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
            hidden_states: List of per-block outputs (empty list if return_hidden_states=False).
        """
        *batch_dims, _ = x.shape
        if sequence_id is None:
            sequence_id = torch.ones(
                size=batch_dims, dtype=torch.int64, device=x.device
            )

        hidden_states = []
        for block in self.blocks:
            x = block(
                x=x,
                sequence_id=sequence_id,
            )
            if return_hidden_states:
                hidden_states.append(x)
        return self.norm(x), x, hidden_states
