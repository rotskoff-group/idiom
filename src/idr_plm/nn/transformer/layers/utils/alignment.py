import torch
from clm.utils.affine import Affine3D


def compute_alignment_tensors(mobile, target, atom_exists_mask, sequence_id):
    """
    Align two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, S, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, S, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, S)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.

    Returns:
    - centered_mobile (torch.Tensor): Batch of coordinates of structure centered mobile (B, S, 3)
    - centroid_mobile (torch.Tensor): Batch of coordinates of mobile centeroid (B, 3)
    - centered_target (torch.Tensor): Batch of coordinates of structure centered target (B, S, 3)
    - centroid_target (torch.Tensor): Batch of coordinates of target centeroid (B, 3)
    - rotation_matrix (torch.Tensor): Batch of coordinates of rotation matrix (B, 3, 3)
    - num_valid_atoms (torch.Tensor): Batch of number of valid atoms for alignment (B,)
    """

    # Ensure both batches have the same number of structures, atoms, and dimensions
    if sequence_id is not None:
        raise NotImplementedError("Sequence ID not supported yet")
        mobile = unbinpack(mobile, sequence_id, pad_value=torch.nan)
        target = unbinpack(target, sequence_id, pad_value=torch.nan)
        if atom_exists_mask is not None:
            atom_exists_mask = unbinpack(atom_exists_mask, sequence_id, pad_value=0)
        else:
            atom_exists_mask = torch.isfinite(target).all(-1)

    assert mobile.shape == target.shape, "Batch structure shapes do not match!"

    # Number of structures in the batch
    batch_size = mobile.shape[0]

    # if [B, Nres, Natom, 3], resize
    if mobile.dim() == 4:
        mobile = mobile.reshape(batch_size, -1, 3)
    if target.dim() == 4:
        target = target.reshape(batch_size, -1, 3)
    if atom_exists_mask is not None and atom_exists_mask.dim() == 3:
        atom_exists_mask = atom_exists_mask.reshape(batch_size, -1)

    # Number of atoms
    num_atoms = mobile.shape[1]

    # Apply masks if provided
    if atom_exists_mask is not None:
        mobile = mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
        target = target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    else:
        atom_exists_mask = torch.ones(
            batch_size, num_atoms, dtype=torch.bool, device=mobile.device
        )

    num_valid_atoms = atom_exists_mask.sum(dim=-1, keepdim=True)
    # Compute centroids for each batch
    centroid_mobile = mobile.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)
    centroid_target = target.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)

    # Handle potential division by zero if all atoms are invalid in a structure
    centroid_mobile[num_valid_atoms == 0] = 0
    centroid_target[num_valid_atoms == 0] = 0

    # Center structures by subtracting centroids
    centered_mobile = mobile - centroid_mobile
    centered_target = target - centroid_target

    centered_mobile = centered_mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    centered_target = centered_target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)

    # Compute covariance matrix for each batch
    covariance_matrix = torch.matmul(centered_mobile.transpose(1, 2), centered_target)

    # Singular Value Decomposition for each batch
    u, _, v = torch.svd(covariance_matrix)

    # Calculate rotation matrices for each batch
    rotation_matrix = torch.matmul(u, v.transpose(1, 2))

    return (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    )


def compute_rmsd_no_alignment(aligned, target, num_valid_atoms, reduction="batch"):
    """
    Compute RMSD between two batches of structures without alignment.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, S, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, S, 3)
    - num_valid_atoms (torch.Tensor): Batch of number of valid atoms for alignment (B,)
    - reduction (str): One of "batch", "per_sample", "per_residue".

    Returns:

    If reduction == "batch":
        (torch.Tensor): 0-dim, Average Root Mean Square Deviation between the structures for each batch
    If reduction == "per_sample":
        (torch.Tensor): (B,)-dim, Root Mean Square Deviation between the structures for each batch
    If reduction == "per_residue":
        (torch.Tensor): (B, N)-dim, Root Mean Square Deviation between the structures for residue in the batch
    """
    if reduction not in ("per_residue", "per_sample", "batch"):
        raise ValueError("Unrecognized reduction: '{reduction}'")
    # Compute RMSD for each batch
    diff = aligned - target
    if reduction == "per_residue":
        mean_squared_error = diff.square().view(diff.shape[0], -1, 9).mean(dim=-1)
    else:
        mean_squared_error = diff.square().sum(dim=(1, 2)) / (
            num_valid_atoms.squeeze(-1) * 3
        )

    rmsd = torch.sqrt(mean_squared_error)
    if reduction in ("per_sample", "per_residue"):
        return rmsd
    elif reduction == "batch":
        avg_rmsd = rmsd.masked_fill(num_valid_atoms.squeeze(-1) == 0, 0).sum() / (
            (num_valid_atoms > 0).sum() + 1e-8
        )
        return avg_rmsd
    else:
        raise ValueError(reduction)


def compute_affine_and_rmsd(mobile, target, atom_exists_mask=None, sequence_id=None):
    """
    Compute RMSD between two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, S, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, S, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, S)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.

    Returns:
    - affine (Affine3D): Transformation between mobile and target structure
    - avg_rmsd (torch.Tensor): Average Root Mean Square Deviation between the structures for each batch
    """

    (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    ) = compute_alignment_tensors(
        mobile=mobile,
        target=target,
        atom_exists_mask=atom_exists_mask,
        sequence_id=sequence_id,
    )

    # Apply rotation to mobile centroid
    translation = torch.matmul(-centroid_mobile, rotation_matrix) + centroid_target
    affine = Affine3D(
        rot=rotation_matrix.unsqueeze(dim=-3).transpose(-2, -1), trans=translation
    )

    # Apply transformation to centered structure to compute rmsd
    rotated_mobile = torch.matmul(centered_mobile, rotation_matrix)
    avg_rmsd = compute_rmsd_no_alignment(
        rotated_mobile,
        centered_target,
        num_valid_atoms,
        reduction="per_sample",
    )
    return affine, avg_rmsd


def compute_rmsd(
    mobile_coords,
    target_coords,
    mobile_mask=None,
    target_mask=None,
    use_reflection=False,
):
    # Check proteins must have same number of residues
    assert len(mobile_coords) == len(target_coords)

    # Determine overlapping atoms
    joint_mask = mobile_mask.bool() & target_mask.bool()

    # If using reflection flip target
    if use_reflection:
        target_coords = -target_coords

        # Compute alignment and rmsd
    affine3D, rmsd = compute_affine_and_rmsd(
        mobile_coords, target_coords, atom_exists_mask=joint_mask
    )
    return affine3D, rmsd
