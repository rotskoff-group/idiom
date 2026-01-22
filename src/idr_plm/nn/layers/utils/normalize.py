import torch
from .affine import graham_schmidt, Affine3D


def get_protein_normalization_frame(bb_coords):
    """Given a set of coordinates for a protein, compute a single frame that can be used to normalize the coordinates.
    Specifically, we compute the average position of the N, CA, and C atoms use those 3 points to construct a frame
    using the Gram-Schmidt algorithm. The average CA position is used as the origin of the frame.

    Args:
        bb_coords (torch.FloatTensor): (B, S, 3, 3) tensor of coordinates

    Returns:
        Affine3D: tensor of Affine3D frame
    """
    # (B, S, 3, 3) -> (B, S)
    coord_mask = torch.all(torch.all(torch.isfinite(bb_coords), dim=-1), dim=-1)

    bb_coords = bb_coords.clone().float()
    bb_coords.masked_fill_(~coord_mask.unsqueeze(-1).unsqueeze(-1), 0)
    average_per_n_ca_c = bb_coords.sum(1) / (
        coord_mask.sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-8
    )

    N, CA, C = average_per_n_ca_c.unbind(dim=-2)
    neg_x_axis = C
    origin = CA
    x_axis = origin - neg_x_axis
    xy_plane = N - origin

    # (B, 3, 3)
    affine_from_average_rot = graham_schmidt(x_axis, xy_plane, eps=1e-12)
    # (B, 3)
    affine_from_average_trans = origin  # CA

    affine = Affine3D(affine_from_average_rot, affine_from_average_trans)

    return affine


def normalize_backbone_coordinates(bb_coords):
    """
    Given a set of coordinates for a protein, compute a single frame that can be used to normalize the coordinates.
    Args:
        bb_coords: (B, S, 3, 3) tensor of coordinates
    """
    normalization_frame = get_protein_normalization_frame(bb_coords)

    # invert normalization frame
    # (B, S, 1, 3, 3)
    normalization_frame_invert_rots = (
        normalization_frame.get_rot().unsqueeze(-3).unsqueeze(-3).transpose(-1, -2)
    )
    # (B, S, 1, 3)
    normalization_frame_invert_trans = -(
        normalization_frame.get_trans().unsqueeze(-2).unsqueeze(-2)
        @ normalization_frame.get_rot().unsqueeze(-3)
    )
    # apply to coords
    # (B, S, 3, 3)
    coords_trans_rot = (
        bb_coords @ normalization_frame_invert_rots.transpose(-1, -2).squeeze(-3)
        + normalization_frame_invert_trans
    )

    # Get valid frames
    # (B, )
    valid_frame = normalization_frame.get_trans().norm(dim=-1) > 0
    # (B, S, 3, 3)
    is_inf = torch.isinf(bb_coords)
    # (B, S, 3, 3)
    bb_coords = coords_trans_rot.where(
        valid_frame.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), bb_coords
    )
    bb_coords.masked_fill_(is_inf, torch.inf)

    return bb_coords
