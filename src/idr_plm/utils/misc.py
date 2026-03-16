from typing import Sequence, TypeVar
import random
import numpy as np
import torch

MAX_SUPPORTED_DISTANCE = 1e6

TSequence = TypeVar("TSequence", bound=Sequence)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def extract_disordered_regions(sequence):
    """Extract only the disordered regions (marked by '2') from a sequence"""
    parts = {}
    current_part = ""
    current_marker = None

    for char in sequence:
        if char in ["1", "2", "3"]:
            if current_marker is not None:
                parts[current_marker] = current_part
            current_marker = char
            current_part = ""
        else:
            current_part += char

    # Add the last part
    if current_marker is not None:
        parts[current_marker] = current_part

    # Return only the disordered region (marked by '2')
    return parts.get("2", "")


def rearrange_sequence(sequence):
    """
    Rearranges a sequence to be in order 1...2...3... without including the markers
    Args:
        sequence (str): Input sequence containing markers 1, 2, and 3
    Returns:
        str: Rearranged sequence without markers
    """
    # Split the sequence into parts
    parts = {}
    current_part = ""
    current_marker = None

    for char in sequence:
        if char in ["1", "2", "3"]:
            if current_marker is not None:
                parts[current_marker] = current_part
            current_marker = char
            current_part = ""
        else:
            current_part += char

    # Add the last part
    if current_marker is not None:
        parts[current_marker] = current_part

    # Combine parts in order 1, 2, 3 (without including the markers)
    rearranged = ""
    for marker in ["1", "2", "3"]:
        if marker in parts:
            rearranged += parts[marker]

    return rearranged


def extract_idr_with_indices(sequence):
    """
    Extract IDR region from marked sequence and return indices.

    Args:
        sequence (str): Input sequence containing markers 1, 2, and 3

    Returns:
        tuple: (idr_sequence, idr_start, idr_end) where start and end are 0-based indices
               in the rearranged sequence. Returns ('', 0, 0) if no IDR found.
    """
    # First, find the IDR region
    idr_seq = extract_disordered_regions(sequence)
    if not idr_seq:
        return "", 0, 0

    # Get the rearranged sequence (without markers)
    # rearranged = rearrange_sequence(sequence)

    # Find the position of IDR in the rearranged sequence
    # It corresponds to everything between '1' and '3' that's marked as '2'
    parts = {}
    current_part = ""
    current_marker = None

    for char in sequence:
        if char in ["1", "2", "3"]:
            if current_marker is not None:
                parts[current_marker] = current_part
            current_marker = char
            current_part = ""
        else:
            current_part += char

    if current_marker is not None:
        parts[current_marker] = current_part

    # Calculate indices in the rearranged sequence
    part1_len = len(parts.get("1", ""))
    idr_start = part1_len
    idr_end = part1_len + len(idr_seq)

    return idr_seq, idr_start, idr_end
