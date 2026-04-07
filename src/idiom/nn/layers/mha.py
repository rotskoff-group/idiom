import torch
import math
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from .rotary import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        bias=False,
        qk_layernorm=True,
        mask_mode="causal",
        cross_attention=False,
    ):
        """Implementation of MultiHeadAttention

        d_model: Model embedding dimension
        num_heads: Number of heads for attention, should divide d_model evenly
        bias: If an additive bias term is added to the layer normalization
        qk_layernorm: If layer norm is applied to the query and key tensors after
            splitting from the initial input
        cross_attention: If True, enables cross-attention mode where key/value can be
            different from query. When False, uses self-attention.
        masked_mode: Controls how the sequence id is used to construct the mask:
            packed_seq: Sequences of the form [a,a,a,b,b,b,...] where same values attend
                to each other. This is the default mode. This mode should be used when training
                BIDIRECTIONAL models.
            causal: Autoregressive lower triangular mask. Sequence id is assumed to be
                binary, i.e. of the form [1,1,1,1,0,0,0,...] where 1 is for valid tokens and
                0 is for padding tokens. This mode should be used when training AUTOREGRESSIVE models.
            transfusion: Autorgressive mask for tokns and all-to-all attention for
                structures. This assumes that:
                - 0: padding
                - 1: token
                - 2: structure
                and sequences have the form [1,1,1,2,2,2,0,0,0,...].
                This mode should be used when training TRANSFUSION models that mix discrete token
                and continuous structure modalities.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.mask_mode = mask_mode
        self.cross_attention = cross_attention

        self.d_head = self.d_model // self.num_heads

        if cross_attention:
            self.q_proj = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model, bias=bias)
            )
            self.kv_proj = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2, bias=bias)
            )
        else:
            self.layernorm_qkv = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
            )

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(self.d_head)

    def _apply_rotary(self, q, k):
        q = q.unflatten(-1, (self.num_heads, self.d_head))
        k = k.unflatten(-1, (self.num_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            attn_mask = attn_mask.squeeze()
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(self, x, sequence_id, key=None, value=None, key_sequence_id=None):
        if self.cross_attention:
            if key is None or value is None:
                raise ValueError("key and value must be provided for cross-attention")
            if key_sequence_id is None:
                raise ValueError("key_sequence_id must be provided for cross-attention")

            query_BLD = self.q_proj(x)
            kv_BLD2 = self.kv_proj(
                key
                if key.shape == value.shape and torch.equal(key, value)
                else torch.cat([key, value], dim=0)
                .mean(dim=0, keepdim=True)
                .expand(key.shape[0], -1, -1)
            )
            key_BLD, value_BLD = torch.chunk(kv_BLD2, 2, dim=-1)
        else:
            if key is not None or value is not None:
                raise ValueError(
                    "key and value should not be provided for self-attention"
                )
            qkv_BLD3 = self.layernorm_qkv(x)  # pre-LayerNorm
            query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
            key_sequence_id = sequence_id

        query_BLD, key_BLD = self.q_ln(query_BLD), self.k_ln(key_BLD)

        # Reshape for rotary embeddings first, more interpretable einops string than
        # the self._apply_rotary() function which uses flattens
        query_BSHD = rearrange(query_BLD, "b s (h d) -> b s h d", h=self.num_heads)
        key_BSHD = rearrange(key_BLD, "b s (h d) -> b s h d", h=self.num_heads)

        # Apply rotary embeddings
        if self.cross_attention:
            # For cross-attention, query and key may have different sequence lengths
            query_BSHD, key_BSHD = self.rotary(
                query_BSHD, key_BSHD, seqlen_offset=0, k_seqlen_offset=0
            )
        else:
            # For self-attention, query and key have the same sequence length
            query_BSHD, key_BSHD = self.rotary(query_BSHD, key_BSHD)

        # Transpose to format expected by scaled_dot_product_attention: b h s d
        query_BHLD = rearrange(query_BSHD, "b s h d -> b h s d")
        key_BHLD = rearrange(key_BSHD, "b s h d -> b h s d")
        value_BHLD = rearrange(value_BLD, "b s (h d) -> b h s d", h=self.num_heads)

        if self.mask_mode == "packed_seq":
            if self.cross_attention:
                # For cross-attention, mask based on query and key sequence IDs
                mask_BLL = sequence_id.unsqueeze(-1) == key_sequence_id.unsqueeze(-2)
            else:
                mask_BLL = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)

            mask_BHLL = mask_BLL.unsqueeze(1)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, attn_mask=mask_BHLL
            )

        elif self.mask_mode == "test":
            if self.cross_attention:
                mask_BLL = sequence_id.unsqueeze(-1) == key_sequence_id.unsqueeze(-2)
            else:
                mask_BLL = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)

            context_BHLD = self.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, attn_mask=mask_BHLL
            )

        elif self.mask_mode == "causal":
            if self.cross_attention:
                # For cross-attention with causal mask, create mask between query and key sequences
                mask_outer = torch.einsum("bi,bj->bij", sequence_id, key_sequence_id)
                # No causal constraint for cross-attention, just padding mask
                mask_BLL = torch.zeros_like(mask_outer).float()
                # Mask out positions where either query or key is padding
                padding_mask = (sequence_id.unsqueeze(-1) == 0) | (
                    key_sequence_id.unsqueeze(-2) == 0
                )
                mask_BLL.masked_fill_(padding_mask, float("-inf"))
                mask_BHLL = mask_BLL.unsqueeze(1)
            else:
                # Construct a mask to use for training AUTOREGRESSIVE models
                # This masking procedure generates a square subsequent mask. For example,
                #   there are 6 elements in the sequence. Then, the square mask will be as follows:
                #   [[0, -inf, -inf, -inf, -inf, -inf],
                #    [0, 0, -inf, -inf, -inf, -inf],
                #    [0, 0, 0, -inf, -inf, -inf],
                #    [0, 0, 0, 0, -inf, -inf],
                #    [0, 0, 0, 0, 0, -inf],
                #    [0, 0, 0, 0, 0, 0]]
                # Note that the lower triangular nature of the mask is enforced over all tokens.
                # Make sure to exclude padding in any downstream loss calculations in cross entropy loss!
                # For more, see detailed SPDA impl at
                #   https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

                # FH: Generate square subsequent mask. Here, sequence_id should have shape (B, T) where
                # B is the batch size and T is the sequence length.
                B, T = sequence_id.shape
                mask_BHLL = torch.triu(
                    # 0 along the main diagonal
                    torch.ones(B, 1, T, T, device=sequence_id.device),
                    diagonal=1,
                )
                mask_BHLL = mask_BHLL.masked_fill_(mask_BHLL.bool(), float("-inf"))

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, attn_mask=mask_BHLL
            )

        context_BLD = rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)
