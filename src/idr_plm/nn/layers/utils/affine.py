import torch

_MAX_SUPPORTED_DISTANCE = 1e6


class Affine3D:
    def __init__(self, rot, trans):
        # (..., 3, 3)
        self.rot = rot
        # (..., 3)
        self.trans = trans

    def get_trans(self):
        # (..., 3)
        return self.trans

    def get_rot(self):
        # (..., 3, 3)
        return self.rot

    def get_T(self):
        # (..., 9)
        rot = self.rot.flatten(-2)
        # (..., 12)
        return torch.cat([rot, self.trans], dim=-1)

    @staticmethod
    def from_frame(T):
        if T.shape[-1] == 12:
            trans = T[..., -3:]
            rot = T[..., :-3].unflatten(-1, (3, 3))
        else:
            raise ValueError("Unspoorted T passed in")
        return Affine3D(rot, trans)


def graham_schmidt(x_axis, xy_plane, eps=1e-12):
    """Args:
        x_axis: (batch_size, S, 3)
        xy_plane: (batch_size, S, 3)
        eps: float
    Returns:
        rots: (batch_size, S, 3, 3)
    """
    e1 = xy_plane

    denom = torch.sqrt((x_axis**2).sum(dim=-1, keepdim=True) + eps)
    x_axis = x_axis / denom

    dot = (x_axis * e1).sum(dim=-1, keepdim=True)
    e1 = e1 - x_axis * dot
    denom = torch.sqrt((e1**2).sum(dim=-1, keepdim=True) + eps)
    e1 = e1 / denom
    e2 = torch.cross(x_axis, e1, dim=-1)

    rots = torch.stack([x_axis, e1, e2], dim=-1)  # (3, 3)
    return rots


def build_affine3d_from_coordinates(coords):
    """
    Args:
        coords: (B, S, 3, 3) tensor of coordinates
    """

    # Mask out coordinates that are large/infinite
    # (B, S)
    coord_mask = torch.all(
        torch.all(torch.isfinite(coords) & (coords < _MAX_SUPPORTED_DISTANCE), dim=-1),
        dim=-1,
    )
    coords = coords.clone().float()
    coords[~coord_mask] = 0

    # (B, S, 3)
    average_per_n_ca_c = coords.sum(1) / (
        coord_mask.sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-8
    )

    # (B, 3)
    N, CA, C = average_per_n_ca_c.unbind(dim=-2)
    neg_x_axis = C
    origin = CA
    x_axis = origin - neg_x_axis
    xy_plane = N - origin

    affine_from_average_rot = graham_schmidt(x_axis, xy_plane, eps=1e-12)
    # (B, 9)
    affine_from_average_rot = affine_from_average_rot.flatten(-2)
    # (B, 3)
    affine_from_average_trans = origin  # CA

    B, S, _, _ = coords.shape
    # (B, S, 9)
    affine_rot_mats = affine_from_average_rot.unsqueeze(1).expand(B, S, 9)
    # (B, S, 3)
    affine_trans = affine_from_average_trans.unsqueeze(1).expand(B, S, 3)

    identity_rots = torch.eye(3).to(coords.device)
    # (1, 1, 3, 3)
    identity_rots = identity_rots.unsqueeze(0).unsqueeze(0)
    # (B, S, 3, 3)
    identity_rots = identity_rots.expand(B, S, -1, -1)
    # Make everything False identity matrix
    affine_rot_mats = affine_rot_mats.where(
        coord_mask.any(-1).unsqueeze(-1).unsqueeze(-1), identity_rots.flatten(-2)
    )

    # (B, S, 12)
    affine_black_hole_tensor = torch.cat([affine_rot_mats, affine_trans], dim=-1)
    # (B, S, 3)
    N, CA, C = coords.unbind(dim=-2)
    neg_x_axis = C
    origin = CA
    x_axis = origin - neg_x_axis
    xy_plane = N - origin

    affine_from_coords_rot = graham_schmidt(x_axis, xy_plane, eps=1e-12)
    # (B, S, 9)
    affine_from_coords_rot = affine_from_coords_rot.flatten(-2)
    # (B, S, 3)
    affine_from_coords_trans = origin

    # (B, S, 12)
    affine_from_coords_tensor = torch.cat(
        [affine_from_coords_rot, affine_from_coords_trans], dim=-1
    )

    # Make everything False identity matrix
    t = affine_from_coords_tensor.where(
        coord_mask.unsqueeze(-1), affine_black_hole_tensor
    )
    # (B, S, 3)
    trans = t[..., -3:]
    # (B, S, 3, 3)
    rot = t[..., :-3].unflatten(-1, (3, 3))

    affine = Affine3D(rot, trans)
    return affine, coord_mask
