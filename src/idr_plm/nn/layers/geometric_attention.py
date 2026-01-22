from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class GeometricAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_vector_messages=1,
        mask_and_zero_frameless=True,
        bias=False,
    ):
        """Approximate implementation:

        ATTN(A, v) := (softmax_j A_ij) v_j
        make_rot_vectors(x) := R(i->g) Linear(x).reshape(..., 3)
        make_vectors(x) := T(i->g) Linear(x).reshape(..., 3)

        v <- make_rot_vectors(x)
        q_dir, k_dir <- make_rot_vectors(x)
        q_dist, k_dist <- make_vectors(x)

        A_ij       <- dot(q_dir_i, k_dir_j) -||q_dist_i - k_dist_j||^2
        x          <- x + Linear(T(g->i) ATTN(A, v))
        """
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.m = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless

        self.s_norm = nn.LayerNorm(d_model, bias=bias)

        # (Q_r, K_r, V_d, K_d) = 4
        # num_vector_messages = V
        # (num_heads * 3 * 5) + (num_heads * 3 * (num_vector_messages - 1))
        dim_proj = (4 + self.m) * (self.num_heads * 3)

        self.proj = nn.Linear(d_model, dim_proj, bias=bias)
        channels_out = self.num_heads * 3 * self.m
        self.out_proj = nn.Linear(channels_out, d_model, bias=bias)

        # The basic idea is for some attention heads to pay more or less attention to rotation versus distance,
        # as well as to control the sharpness of the softmax (i.e., should this head only attend to those residues
        # very nearby or should there be shallower dropoff in attention weight?)
        self.distance_scale_per_head = nn.Parameter(torch.zeros((self.num_heads)))
        self.rotation_scale_per_head = nn.Parameter(torch.zeros((self.num_heads)))

    def forward(self, x, affine, affine_mask, sequence_id):
        """
        Args:
            x: (B, S, C) tensor of structure features
            affine: Affine3D class that contains the following:
                trans: (B, S, 3) tensor of affine translations
                rot: (B, S, 3, 3) tensor of affine rotations
            affine_mask: (B, S) tensor of masks for the affine transformations
            sequence_id: (B, S) tensor of sequence ids
        """
        attn_bias = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
        attn_bias = attn_bias.unsqueeze(1).float()
        attn_bias = attn_bias.masked_fill(
            ~affine_mask.unsqueeze(-2).unsqueeze(-2), torch.finfo(attn_bias.dtype).min
        )

        trans, rot = affine.get_trans(), affine.get_rot()

        # (B, S, C) -> (B, S, C)
        nx = self.s_norm(x)
        # Get QKV for both rot and dist
        # (B, S, C) -> (B, S, (2 + M) * 3 * H), (B, S, 2 * 3 * H)
        vec_rot, vec_dist = self.proj(nx).split(
            [
                (2 + self.m) * self.num_heads * 3,
                2 * self.num_heads * 3,
            ],
            dim=-1,
        )
        # Apply rotation matrix to (Q_r, K_r, V)
        qkv_rot_combined = vec_rot.reshape(*vec_rot.shape[:-1], -1, 3) @ rot.transpose(
            -1, -2
        )

        query_rot, key_rot, value = qkv_rot_combined.split(
            [self.num_heads, self.num_heads, self.num_heads * self.m], dim=-2
        )

        # Apply rotation and translation to (Q_d, K_d)
        qk_dist_combined = (
            vec_dist.reshape(*vec_dist.shape[:-1], -1, 3) @ rot.transpose(-1, -2)
        ) + trans.unsqueeze(-2)
        query_dist, key_dist = qk_dist_combined.chunk(2, dim=-2)

        # d = 3
        query_dist = rearrange(query_dist, "b s h d -> b h s 1 d")
        key_dist = rearrange(key_dist, "b s h d -> b h 1 s d")
        query_rot = rearrange(query_rot, "b s h d -> b h s d")
        key_rot = rearrange(key_rot, "b s h d -> b h d s")
        value = rearrange(value, "b s (h m) d -> b h s (m d)", m=self.m)

        # Get distance at sequence level
        # (B, H, S, S)
        distance_term = (query_dist - key_dist).norm(dim=-1) / sqrt(3)

        # Get dot product at sequence level
        # (B, H, S, S)
        rotation_term = query_rot.matmul(key_rot) / sqrt(3)

        distance_term_weight = (
            F.softplus(self.distance_scale_per_head).unsqueeze(-1).unsqueeze(-1)
        )
        rotation_term_weight = (
            F.softplus(self.rotation_scale_per_head).unsqueeze(-1).unsqueeze(-1)
        )

        attn_weight = (
            rotation_term * rotation_term_weight - distance_term * distance_term_weight
        )

        if attn_bias is not None:
            s_q = attn_weight.size(2)
            s_k = attn_weight.size(3)
            _s_q = max(0, attn_bias.size(2) - s_q)
            _s_k = max(0, attn_bias.size(3) - s_k)
            attn_bias = attn_bias[:, :, _s_q:, _s_k:]
            attn_weight = attn_weight + attn_bias

        attn_weight = torch.softmax(attn_weight, dim=-1)

        # (B, H, S, (M, D))
        attn_out = attn_weight.matmul(value)

        attn_out = attn_out.reshape(*attn_out.shape[:3], self.m, -1)
        attn_out = attn_out.permute(0, 2, 1, 3, 4)
        attn_out = attn_out.flatten(start_dim=-3, end_dim=-2)
        attn_out = attn_out @ rot  # Inverse rotation only

        attn_out = attn_out.flatten(start_dim=-2, end_dim=-1)

        if self.mask_and_zero_frameless:
            attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)

        # (B, S, D_out)
        s = self.out_proj(attn_out)

        return s
