import torch
import torch.nn as nn


# from esm.utils.structure.affine3d import (
#     Affine3D,
#     RotationMatrix,
# )

from clm.utils.affine import Affine3D, graham_schmidt

BB_COORDINATES = [
    [0.5256, 1.3612, 0.0000],
    [0.0000, 0.0000, 0.0000],
    [-1.5251, 0.0000, 0.0000],
]

# class Dim6RotStructureHead(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  trans_scale_factor=10):
#         super().__init__()
#         self.ffn1 = nn.Linear(input_dim, input_dim)
#         self.activation_fn = nn.GELU()
#         self.norm = nn.LayerNorm(input_dim)
#         self.proj = nn.Linear(input_dim, 9 + 7 * 2)
#         self.trans_scale_factor = trans_scale_factor
#         self.bb_local_coords = torch.tensor(BB_COORDINATES).float()

#     def forward(self, x, affine, affine_mask):
#         if affine is None:
#             rigids = Affine3D.identity(
#                 x.shape[:-1],
#                 dtype=x.dtype,
#                 device=x.device,
#                 requires_grad=self.training,
#                 rotation_type=RotationMatrix,
#             )
#         else:
#             rigids = affine

#         # Projection head predicts 3 3-D vectors per residue
#         # Translation vector, and 2 vectors (x, y) that define the local frame
#         # Also predicts the unnomralized sine and cosine components of up to 7 dihedral angles (not used)
#         x = self.ffn1(x)
#         x = self.activation_fn(x)
#         x = self.norm(x)
#         trans, x, y, _ = self.proj(x).split([3, 3, 3, 7 * 2], dim=-1)
#         trans = trans * self.trans_scale_factor
#         x = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
#         y = y / (y.norm(dim=-1, keepdim=True) + 1e-5)

#         # Use graham schmidt to convert trans, x, y into frames T (SE(3))
#         # Convert to local frames and compose them
#         update = Affine3D.from_graham_schmidt(x + trans, trans, y + trans)
#         rigids = rigids.compose(update.mask(affine_mask))
#         affine = rigids.tensor

#         # We approximate the positions of the backbone atoms in the global frame by applying the rigid
#         # transformation to the mean of the backbone atoms in the local frame.
#         all_bb_coords_local = (self.bb_local_coords[None, None, :, :]
#                                .expand(*x.shape[:-1], 3, 3)
#                                .to(x.device))
#         pred_xyz = rigids.unsqueeze(-1).apply(all_bb_coords_local)

#         return affine, pred_xyz


class Dim6RotStructureHead(nn.Module):
    def __init__(self, input_dim, trans_scale_factor=10):
        super().__init__()
        self.ffn1 = nn.Linear(input_dim, input_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, 9 + 7 * 2)
        self.trans_scale_factor = trans_scale_factor
        self.bb_local_coords = torch.tensor(BB_COORDINATES).float()

    def forward(self, x, affine, affine_mask):
        """
        x: (B, S, embed_dim)
        affine: Affine3D
        affin_mask: (B, S)
        """
        if affine is None:
            ### Obtain the Affine3D identity ###
            leading_shape = x.shape[:-1]
            # Translation identity
            trans = torch.zeros(
                (*leading_shape, 3),
                dtype=x.dtype,
                device=x.device,
                requires_grad=self.training,
            )
            # Rotation identity
            rot = torch.eye(
                3, dtype=x.dtype, device=x.device, requires_grad=self.training
            )
            rot = rot.view(*[1 for _ in range(len(leading_shape))], 3, 3)
            rot = rot.expand(*leading_shape, -1, -1)
            # Identity Affine3D
            rigids = Affine3D(rot, trans)
        else:
            rigids = affine
        # torch.save(rigids.trans, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/rigids_trans.pt")
        # torch.save(rigids.rot, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/rigids_rots.pt")

        # [*, N]
        # Projection head predicts 3 3-D vectors per residue
        # Translation vector, and 2 vectors (x, y) that define the local frame
        # Also predicts the unnomralized sine and cosine components of up to 7 dihedral angles (not used)
        x = self.ffn1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        trans, x, y, _ = self.proj(x).split([3, 3, 3, 7 * 2], dim=-1)
        trans = trans * self.trans_scale_factor
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-5)
        # x, y, trans: (N, S, 3)

        # torch.save(x, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/x.pt")
        # torch.save(y, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/y.pt")
        # torch.save(trans, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/trans.pt")
        ### Compute the update ###

        # Graham schmidt application
        x_axis = trans - (x + trans)
        xy_plane = (y + trans) - trans
        # (N, S, 3, 3)
        rots_update = graham_schmidt(x_axis, xy_plane)
        # (N, S, 3)
        trans_update = trans

        # torch.save(rots_update, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/update_rots.pt")
        # torch.save(trans_update, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/update_trans.pt")

        ### Composition step ###

        # First, mask the update
        update_shape = trans_update.shape[:-1]
        update_identity_rot = torch.eye(3, device=x.device, dtype=x.dtype)
        update_identity_rot = update_identity_rot.view(
            *[1 for _ in range(len(update_shape))], 3, 3
        )
        update_identity_rot = update_identity_rot.expand(*update_shape, 3, 3)
        update_identity_trans = torch.zeros(
            (*update_shape, 3), device=x.device, dtype=x.dtype
        )
        # (N, S, 12)
        update_identity_tensor = torch.cat(
            [update_identity_rot.flatten(-2), update_identity_trans], dim=-1
        )

        # (N, S, 12)
        masked_update_tensor = update_identity_tensor.where(
            affine_mask.unsqueeze(-1),
            torch.cat([rots_update.flatten(-2), trans_update], dim=-1),
        )

        # Split back out rotation and translation
        rot_update = masked_update_tensor[..., :-3].unflatten(-1, (3, 3))
        trans_update = masked_update_tensor[..., -3:]

        # torch.save(trans_new_update, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/masked_update_trans.pt")
        # torch.save(
        #     rot_new_update, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/masked_update_rots.pt")

        rigids_trans, rigids_rot = rigids.get_trans(), rigids.get_rot()

        rot_compose = rigids_rot @ rot_update
        trans_compose = (
            torch.einsum("...ij,...j", rigids_rot, trans_update) + rigids_trans
        )

        # torch.save(rot_compose, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/compose_rots.pt")
        # torch.save(trans_compose, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/compose_trans.pt")

        affine = torch.cat([rot_compose.flatten(-2), trans_compose], dim=-1)

        # We approximate the positions of the backbone atoms in the global frame by applying the rigid
        # transformation to the mean of the backbone atoms in the local frame.
        # all_bb_coords_local = (self.bb_local_coords[None, None, :, :]
        #                        .expand(*x.shape[:-1], 3, 3)
        #                        .to(x.device))
        # Rewrote above using unsqueeze instead
        # (N, S, 3, 3)
        all_bb_coords_local = (
            self.bb_local_coords.unsqueeze(0)
            .unsqueeze(0)
            .expand(*x.shape[:-1], 3, 3)
            .to(x.device)
        )

        # (N, S, 3, 3)
        pred_xyz = (
            all_bb_coords_local
            @ rot_compose.unsqueeze(-3).transpose(-1, -2).squeeze(-3)
        ) + trans_compose.unsqueeze(-2)

        # torch.save(all_bb_coords_local, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/all_bb_coords_local.pt")
        # torch.save(pred_xyz, "/scratch/group_scratch/esm/070324TestVQVAE/outputs/pred_xyz.pt")

        return affine, pred_xyz
