"""
Define your custom reward functions here. Your reward function name must start with "compute_" and take tokens, token_info, and device as args. Please See compute_fraction_proline() as an example.
"""

import torch


def compute_fraction_proline(tokens, token_info, device):
    """
    Compute the fraction of proline residues in a protein sequence.

    Uses token_info["alphabet"] to convert tokens to amino acid sequence.
    Returns the fraction of proline (P) residues in the sequence.
    """

    _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
    _stop_token = token_info["input"]["TOK"]["TOK_STOP"]
    _start_token = token_info["input"]["TOK"]["TOK_START"]

    # Extract valid tokens (skip special tokens)
    valid_tokens = tokens[
        (tokens != _pad_token) & (tokens != _start_token) & (tokens != _stop_token)
    ]

    if len(valid_tokens) == 0:
        return torch.tensor(0.0, device=device)  # No valid tokens

    # Convert to amino acid sequence using alphabet from token_info
    if "alphabet" in token_info and token_info["alphabet"] is not None:
        alphabet = token_info["alphabet"]
        # Decode bytes alphabet
        alphabet = [item.decode("utf-8") for item in alphabet]
        sequence = "".join([alphabet[token.item()] for token in valid_tokens])
    else:
        # Fallback: return 0 if no alphabet available
        print(
            "Warning: No alphabet found in token_info, cannot convert tokens to sequence"
        )
        return torch.tensor(0.0, device=device)

    # Count proline residues
    proline_count = sequence.upper().count("P")
    total_residues = len(sequence) - 3  # -3 for 1,2,3 FIM sentinels

    if total_residues == 0:
        return torch.tensor(0.0, device=device)

    fraction_proline = proline_count / total_residues
    return torch.tensor(fraction_proline, device=device)
