"""
Define your custom reward functions here. Your reward function name must start with "compute_" and take tokens, token_info, and device as args. Please See compute_fraction_proline() as an example.
"""

import torch
from idr_plm.nn.transformer.scores import extract_disordered_regions
from idr_plm.utils.misc import tokens_to_sequence, rearrange_sequence


def compute_fraction_proline(tokens, token_info, device):
    """
    Compute the fraction of proline residues in a protein sequence.

    Uses token_info["alphabet"] to convert tokens to amino acid sequence.
    Returns the fraction of proline (P) residues in the sequence.
    """

    # Full generated sequence, including N-terminal prefix, C-terminal suffix, and other tokens
    generated_fim_sequence = tokens_to_sequence(tokens, token_info)

    # Full length protein sequence, with correct order and prefix, suffix, IDR span tokens removed
    sequence = rearrange_sequence(generated_fim_sequence)

    # Extract disordered region sequence
    disordered_region, _, _ = extract_disordered_regions(sequence)

    # Count proline residues in disordered region
    proline_count = disordered_region.upper().count("P")
    total_residues = len(disordered_region)

    fraction_proline = proline_count / total_residues
    return torch.tensor(fraction_proline, device=device)
