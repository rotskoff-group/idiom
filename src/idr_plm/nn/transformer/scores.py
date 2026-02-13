import torch
import sys
import os
import pickle
from argparse import Namespace
import random
from Bio import pairwise2

# Global variables for ProtGPS model caching
_PROTGPS_MODEL = None
_PROTGPS_DEVICE = None

COMPARTMENT_CLASSES = [
    "nuclear_speckle",
    "p-body",
    "pml-bdoy",
    "post_synaptic_density",
    "stress_granule",
    "chromosome",
    "nucleolus",
    "nuclear_pore_complex",
    "cajal_body",
    "rna_granule",
    "cell_junction",
    "transcriptional",
]


def compute_fraction_alanine(tokens, token_info, device):
    """
    Compute the fraction of alanine residues in a protein sequence.

    Uses token_info["alphabet"] to convert tokens to amino acid sequence.
    Returns the fraction of alanine (A) residues in the sequence.
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

    # Count alanine residues
    alanine_count = sequence.upper().count("A")
    total_residues = len(sequence) - 3  # -3 for 1,2,3 FIM sentinels

    if total_residues == 0:
        return torch.tensor(0.0, device=device)

    fraction_alanine = alanine_count / total_residues
    return torch.tensor(fraction_alanine, device=device)


def load_protgps_model(protgps_parent_dir="/home/protgps", device=None):
    """
    Load the ProtGPS model for predicting protein localization.

    Args:
        protgps_parent_dir: Path to the ProtGPS parent directory
        device: PyTorch device to load the model on

    Returns:
        model: Loaded ProtGPS model
    """
    global _PROTGPS_MODEL, _PROTGPS_DEVICE

    # Return cached model if already loaded on the same device
    if _PROTGPS_MODEL is not None and _PROTGPS_DEVICE == device:
        return _PROTGPS_MODEL

    # Add ProtGPS to path
    if protgps_parent_dir not in sys.path:
        sys.path.append(protgps_parent_dir)

    from protgps.utils.loading import get_object

    # Load model args and checkpoint
    args_path = os.path.join(
        protgps_parent_dir, "checkpoints/protgps/32bf44b16a4e770a674896b81dfb3729.args"
    )
    ckpt_path = os.path.join(
        protgps_parent_dir,
        "checkpoints/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt",
    )

    args = Namespace(**pickle.load(open(args_path, "rb")))
    args.model_path = ckpt_path
    args.pretrained_hub_dir = os.path.join(protgps_parent_dir, "esm_models/esm2")

    # Load model
    model = get_object(args.lightning_name, "lightning")(args)
    model = model.load_from_checkpoint(
        checkpoint_path=args.model_path,
        strict=not args.relax_checkpoint_matching,
        **{"args": args},
    )

    model.eval()
    if device is not None:
        model = model.to(device)

    # Cache the model
    _PROTGPS_MODEL = model
    _PROTGPS_DEVICE = device

    return model


def compute_protgps_score(
    tokens,
    token_info,
    device,
    target_compartment="p-body",
    protgps_parent_dir="/home/protgps",
    aggregation="max",
):
    """
    Compute ProtGPS condensate localization score for a protein sequence.

    Uses token_info["alphabet"] to convert tokens to amino acid sequence.
    Returns the probability score for the target compartment (0.0 to 1.0).

    Args:
        tokens: Tensor of token indices
        token_info: Dictionary containing alphabet and special tokens
        device: PyTorch device
        target_compartment: Name of the target compartment to score (default: "p-body")
            Options: "nuclear_speckle", "p-body", "pml-bdoy", "post_synaptic_density",
                     "stress_granule", "chromosome", "nucleolus", "nuclear_pore_complex",
                     "cajal_body", "rna_granule", "cell_junction", "transcriptional"
        protgps_parent_dir: Path to ProtGPS installation directory
        aggregation: How to aggregate multiple compartment scores ("max", "mean", "sum")
            - "max": return max score across all compartments
            - "mean": return mean score across all compartments
            - "sum": return sum of scores across all compartments
            - single compartment name: return score for that compartment only

    Returns:
        Tensor with the ProtGPS score (0.0 to 1.0)
    """
    _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
    _stop_token = token_info["input"]["TOK"]["TOK_STOP"]
    _start_token = token_info["input"]["TOK"]["TOK_START"]

    # Extract valid tokens (skip special tokens)
    valid_tokens = tokens[
        (tokens != _pad_token) & (tokens != _start_token) & (tokens != _stop_token)
    ]

    # Convert to amino acid sequence using alphabet from token_info
    if "alphabet" in token_info and token_info["alphabet"] is not None:
        alphabet = token_info["alphabet"]
        # Decode bytes alphabet
        alphabet = [item.decode("utf-8") for item in alphabet]
        sequence = "".join([alphabet[token.item()] for token in valid_tokens])
    else:
        print(
            "Warning: No alphabet found in token_info, cannot convert tokens to sequence"
        )
        return torch.tensor(0.0, device=device)

    # Extract IDR sequence (region marked by '2')
    idr_sequence, _, _ = extract_disordered_regions(sequence)

    if len(idr_sequence) == 0:
        print("Warning: No IDR sequence found")
        return torch.tensor(0.0, device=device)

    # Load ProtGPS model (will use cached version if available)
    model = load_protgps_model(protgps_parent_dir=protgps_parent_dir, device=device)

    # Predict with ProtGPS
    with torch.no_grad():
        out = model.model({"x": [idr_sequence]})
        scores = torch.sigmoid(out["logit"]).squeeze(
            0
        )  # Shape: (12,) for 12 compartments

    # Return score based on aggregation method
    if aggregation == "max":
        score = scores.max()
    elif aggregation == "mean":
        score = scores.mean()
    elif aggregation == "sum":
        score = scores.sum()
    elif aggregation in COMPARTMENT_CLASSES:
        # Return score for specific compartment
        compartment_idx = COMPARTMENT_CLASSES.index(aggregation)
        score = scores[compartment_idx]
    else:
        # Default to target_compartment
        if target_compartment not in COMPARTMENT_CLASSES:
            print(
                f"Warning: Unknown compartment '{target_compartment}', using 'p-body'"
            )
            target_compartment = "p-body"
        compartment_idx = COMPARTMENT_CLASSES.index(target_compartment)
        score = scores[compartment_idx]

    return score.to(device)


def extract_disordered_regions(sequence):
    """
    Extract the amino acids corresponding to regions marked by '1', '2', and '3' from a sequence.
    Returns a tuple: (disordered_region, region1, region3)
    """
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

    # Return the regions: (disordered, region1, region3)
    return (parts.get("2", ""), parts.get("1", ""), parts.get("3", ""))


def apply_quadratic_reward_shaping(raw_reward, target_value, reward_scale=1.0):
    """
    Apply quadratic reward shaping around a target value.

    The shaped reward is: -reward_scale * (raw_reward - target_value)^2

    This creates a quadratic penalty that is maximized when raw_reward = target_value,
    and decreases quadratically as raw_reward moves away from the target.

    Args:
        raw_reward: The original reward value (tensor or float)
        target_value: The target value we want the reward to be close to (tensor or float)
        reward_scale: Scaling factor for the quadratic penalty (default=1.0)

    Returns:
        Shaped reward value (same type as raw_reward)
    """

    if isinstance(raw_reward, torch.Tensor):
        shaped_reward = -reward_scale * torch.pow(raw_reward - target_value, 2)
    else:
        shaped_reward = -reward_scale * (raw_reward - target_value) ** 2

    return shaped_reward


def compute_length_reward(
    tokens, token_info, device, target_length=100, length_reward_width=1.0
):
    """
    Compute a normalized reward based on sequence length deviation from target.

    The reward uses a quadratic penalty function that can be widened via the length_reward_width parameter:
    reward = -((length - target_length) / (target_length * length_reward_width))^2

    Higher length_reward_width values create a wider penalty curve

    Args:
        tokens: Tensor of token indices
        token_info: Dictionary containing alphabet and special tokens
        device: PyTorch device
        target_length: Target sequence length to optimize toward (default=100)
        length_reward_width: Width factor for the reward curve (default=1.0).
                            Increase to allow more variability around target length.

    Returns:
        Tensor with the normalized length reward (range approximately -1 to 0)
    """
    try:
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
        _stop_token = token_info["input"]["TOK"]["TOK_STOP"]
        _start_token = token_info["input"]["TOK"]["TOK_START"]

        # Extract valid tokens (skip special tokens)
        valid_tokens = tokens[
            (tokens != _pad_token) & (tokens != _start_token) & (tokens != _stop_token)
        ]

        if len(valid_tokens) == 0:
            return torch.tensor(
                -1.0, device=device
            )  # Maximum penalty for empty sequence

        # Convert to amino acid sequence using alphabet from token_info
        if "alphabet" in token_info and token_info["alphabet"] is not None:
            alphabet = token_info["alphabet"]
            # Decode bytes alphabet
            alphabet = [
                item.decode("utf-8") if isinstance(item, bytes) else item
                for item in alphabet
            ]
            sequence = "".join([alphabet[token.item()] for token in valid_tokens])
        else:
            print(
                "Warning: No alphabet found in token_info, cannot convert tokens to sequence"
            )
            return torch.tensor(-1.0, device=device)

        # Extract IDR sequence (region marked by '2')
        idr_sequence, _, _ = extract_disordered_regions(sequence)

        if len(idr_sequence) == 0:
            print("Warning: No IDR sequence found for length reward")
            return torch.tensor(-1.0, device=device)

        # Calculate sequence length
        seq_length = len(idr_sequence)

        # Compute normalized quadratic reward with configurable width parameter
        # Wider parameter allows more deviation while still maintaining gradient toward target
        normalized_deviation = (seq_length - target_length) / (
            target_length * length_reward_width
        )
        length_reward = -(normalized_deviation**2)

        return torch.tensor(length_reward, device=device, dtype=torch.float32)

    except Exception as e:
        print(f"Error computing length reward: {e}")
        return torch.tensor(
            -1.0, device=device
        )  # Return maximum penalty for exceptions


def percent_identity(
    seq1_or_tokens1, seq2_or_tokens2, token_info=None, global_align=True
):
    """
    Calculate % identity between two sequences using pairwise alignment.

    Can accept either:
    1. Two string sequences (if token_info is None)
    2. Two token sequences with token_info provided for conversion to strings

    Args:
        seq1_or_tokens1: Either a string sequence or a token tensor
        seq2_or_tokens2: Either a string sequence or a token tensor
        token_info: Dictionary with token_info (required if inputs are tokens)
        global_align: Whether to use global alignment (True) or local (False)

    Returns:
        Identity percentage as a float (0-100).
        Identity is calculated as matches / min(len(seq1), len(seq2)) * 100.
    """
    # Convert tokens to strings if token_info is provided
    if token_info is not None:
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
        _stop_token = token_info["input"]["TOK"]["TOK_STOP"]
        _start_token = token_info["input"]["TOK"]["TOK_START"]

        try:
            alphabet = token_info["alphabet"]
            alphabet = [
                item.decode("utf-8") if isinstance(item, bytes) else item
                for item in alphabet
            ]
        except (KeyError, TypeError):
            return 0.0

        # Convert first sequence from tokens to string
        tokens1 = seq1_or_tokens1
        valid_tokens1 = tokens1[
            (tokens1 != _pad_token)
            & (tokens1 != _start_token)
            & (tokens1 != _stop_token)
        ]
        if len(valid_tokens1) == 0:
            return 0.0
        seq1 = "".join([alphabet[token.item()] for token in valid_tokens1])

        # Convert second sequence from tokens to string
        tokens2 = seq2_or_tokens2
        valid_tokens2 = tokens2[
            (tokens2 != _pad_token)
            & (tokens2 != _start_token)
            & (tokens2 != _stop_token)
        ]
        if len(valid_tokens2) == 0:
            return 0.0
        seq2 = "".join([alphabet[token.item()] for token in valid_tokens2])
    else:
        # Assume inputs are already strings
        seq1 = seq1_or_tokens1
        seq2 = seq2_or_tokens2

    # Extract disordered region after '2' marker
    if not seq1 or not seq2:
        return 0.0

    # Extract disordered regions (region marked by '2') from both sequences
    seq1_disordered, _, _ = extract_disordered_regions(seq1)
    seq2_disordered, _, _ = extract_disordered_regions(seq2)

    # Use the disordered regions for alignment
    seq1 = seq1_disordered
    seq2 = seq2_disordered

    if not seq1 or not seq2:
        return 0.0

    if global_align:
        alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    else:
        alignments = pairwise2.align.localxx(seq1, seq2, one_alignment_only=True)

    if not alignments:
        return 0.0

    best_alignment = alignments[0]
    aligned1, aligned2, score, start, end = best_alignment

    matches = sum(res1 == res2 for res1, res2 in zip(aligned1, aligned2))

    # Choose normalization based on alignment type:
    # - Global alignment: normalize by alignment length (max of the two sequences)
    # - Local alignment: normalize by min_length (only the aligned region matters)
    if global_align:
        # For global alignment, use the length of the alignment (which includes gaps)
        alignment_length = len(aligned1)
        identity = matches / alignment_length * 100
    else:
        # For local alignment, use the shorter sequence length
        alignment_length = min(len(seq1), len(seq2))
        identity = matches / alignment_length * 100

    return identity


def calculate_percent_identities(
    generated_sequences, token_info, global_align=True, sample_fraction=0.25
):
    """
    Calculate all-to-all percent identities for a list of generated sequences. Mainly used as a diagnostic and not in the reward or loss

    Args:
        generated_sequences: List of generated sequence dictionaries containing 'sequence'
        token_info: Dictionary containing token information including alphabet
        global_align: Whether to use global alignment (True) or local (False)
        sample_fraction: Fraction of sequences to sample for PID calculation (0.0 to 1.0, default=0.25) If < 1.0, randomly samples a subset of sequences before calculating PIDs

    Returns:
        list: List of percent identity values for all unique pairs
    """

    percent_identities = []
    sequences_list = [seq_data["sequence"] for seq_data in generated_sequences]

    # Sample a subset of sequences if sample_fraction < 1.0
    if sample_fraction < 1.0 and len(sequences_list) > 1:
        num_to_sample = max(
            2, int(len(sequences_list) * sample_fraction)
        )  # At least 2 sequences
        sampled_indices = random.sample(range(len(sequences_list)), num_to_sample)
        sequences_list = [sequences_list[i] for i in sampled_indices]

    if len(sequences_list) > 1:
        for i in range(len(sequences_list)):
            for j in range(i + 1, len(sequences_list)):
                pid = percent_identity(
                    sequences_list[i],
                    sequences_list[j],
                    token_info=token_info,
                    global_align=global_align,
                )
                percent_identities.append(pid)

    return percent_identities


def print_example_sequences(
    generated_sequences,
    rewards_tensor,
    raw_rewards_tensor,
    token_info,
    global_step,
    num_examples=5,
):
    """
    Print randomly selected example sequences with their rewards throughout GRPO training.

    Args:
        generated_sequences: List of generated sequence dictionaries containing 'sequence', 'prompt_idx'
        rewards_tensor: Tensor of shaped rewards
        raw_rewards_tensor: Tensor of raw (unshaped) rewards
        token_info: Dictionary containing token information including alphabet
        global_step: Current training step for logging
        num_examples: Number of example sequences to print (default=5)
    """
    import random

    print("=" * 60)
    print(f"Step {global_step}: Example Generated Sequences (randomly selected)")

    num_sequences = len(generated_sequences)
    if num_sequences > 0:
        # Get special tokens
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
        _stop_token = token_info["input"]["TOK"]["TOK_STOP"]
        _start_token = token_info["input"]["TOK"]["TOK_START"]

        # Select up to num_examples random indices
        num_to_print = min(num_examples, num_sequences)
        random_indices = random.sample(range(num_sequences), num_to_print)

        # Get alphabet from token_info for decoding
        try:
            alphabet = token_info["alphabet"]
            # Decode bytes alphabet
            alphabet = [item.decode("utf-8") for item in alphabet]
        except (KeyError, TypeError):
            alphabet = None

        for idx, seq_idx in enumerate(random_indices, 1):
            seq_data = generated_sequences[seq_idx]
            sequence_tensor = seq_data["sequence"]
            reward = rewards_tensor[seq_idx].item()
            raw_reward = raw_rewards_tensor[seq_idx].item()

            # Extract valid tokens (skip special tokens)
            valid_tokens = sequence_tensor[
                (sequence_tensor != _pad_token)
                & (sequence_tensor != _start_token)
                & (sequence_tensor != _stop_token)
            ]

            if len(valid_tokens) == 0 or alphabet is None:
                sequence_str = "[Empty or no alphabet]"
            else:
                # Decode tokens to sequence string
                sequence_str = "".join(
                    [alphabet[token.item()] for token in valid_tokens]
                )

            print(f"\nExample {idx}:")
            print(f"  Prompt Index: {seq_data['prompt_idx']}")
            print(f"  Raw Reward: {raw_reward:.2f}")
            print(f"  Shaped Reward: {reward:.2f}")
            print(f"  Sequence: {sequence_str}")

    print("=" * 60)


def compute_sequence_entropy(sequence, token_info, device):
    """Calculate Shannon entropy of amino acid distribution in the disordered region.

    Extracts the disordered region (marked by '2') and computes entropy only over
    amino acids (ACDEFGHIKLMNPQRSTVWY) in that region, excluding structural markers.

    Args:
        sequence: torch.Tensor of token indices
        token_info: Dictionary containing token information including alphabet
        device: torch device

    Returns:
        torch.Tensor: Shannon entropy value (scalar)
    """
    try:
        pad_token = token_info["input"]["TOK"]["TOK_PAD"]
        start_token = token_info["input"]["TOK"]["TOK_START"]
        stop_token = token_info["input"]["TOK"]["TOK_STOP"]

        # Filter out special tokens (pad, start, stop)
        special_tokens = {pad_token, start_token, stop_token}
        filtered_tokens = sequence[
            ~torch.isin(sequence, torch.tensor(list(special_tokens), device=device))
        ]

        if len(filtered_tokens) == 0:
            return torch.tensor(0.0, device=device)

        # Convert tokens to amino acid string
        if "alphabet" in token_info and token_info["alphabet"] is not None:
            alphabet = token_info["alphabet"]
            alphabet = [
                item.decode("utf-8") if isinstance(item, bytes) else item
                for item in alphabet
            ]
            sequence_str = "".join([alphabet[int(token)] for token in filtered_tokens])
        else:
            return torch.tensor(0.0, device=device)

        # Extract the disordered region (marked by '2')
        disordered_region, _, _ = extract_disordered_regions(sequence_str)

        if len(disordered_region) == 0:
            return torch.tensor(0.0, device=device)

        # Convert disordered region to token indices for counting
        aa_indices = [alphabet.index(char) for char in disordered_region]
        aa_tokens = torch.tensor(aa_indices, device=device)

        # Count token frequencies
        unique_tokens, counts = torch.unique(aa_tokens, return_counts=True)

        # Calculate probabilities
        probabilities = counts.float() / aa_tokens.shape[0]

        # Calculate Shannon entropy: H = -sum(p * log(p))
        entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum()

        return entropy

    except Exception as e:
        print(f"Error computing sequence entropy: {e}")
        return torch.tensor(0.0, device=device)


def compute_entropy_reward(
    tokens, token_info, device, target_entropy=2.75, entropy_reward_width=1.0
):
    """Compute a normalized reward based on sequence entropy deviation from target.

    The reward uses a quadratic penalty function that can be widened via the entropy_reward_width parameter:
    reward = -((entropy - target_entropy) / (target_entropy * entropy_reward_width))^2

    Higher entropy_reward_width values create a wider/softer penalty.

    Args:
        tokens: Tensor of token indices
        token_info: Dictionary containing alphabet and special tokens
        device: PyTorch device
        target_entropy: Target sequence entropy to optimize toward (default=2.5)
        entropy_reward_width: Width factor for the reward curve (default=1.0).
                             Increase to allow more variability around target entropy.

    Returns:
        Tensor with the normalized entropy reward (range approximately -1 to 0)
    """
    try:
        # Compute sequence entropy
        entropy = compute_sequence_entropy(tokens, token_info, device)

        # Calculate normalized deviation from target
        normalized_deviation = (entropy - target_entropy) / (
            target_entropy * entropy_reward_width
        )

        # Quadratic penalty (maximum reward is 0 when entropy = target_entropy)
        reward = -(normalized_deviation**2)

        return reward

    except Exception as e:
        print(f"Error computing entropy reward: {e}")
        return torch.tensor(0.0, device=device)  # Return 0 for exceptions


def valid_sequence_characters(tokens, token_info, device):
    """
    Validates a sequence based on marker rules and amino acid composition.

    Converts token sequence to amino acid string and checks validity.
    Rules for a valid sequence:
    1. Must start with '1' or '3'.
    2. Must contain exactly one '1'.
    3. Must contain exactly one '2'.
    4. Must contain exactly one '3'.
    5. Marker '1' must appear before marker '2'.
    6. Marker '3' must appear before marker '2'.
    7. All characters other than '1', '2', '3' must be standard amino acids
       (ACDEFGHIKLMNPQRSTVWY).

    Args:
        tokens: Tensor of token indices
        token_info: Dictionary containing token information including alphabet and special tokens
        device: PyTorch device

    Returns:
        True if sequence is valid, False otherwise.
    """
    try:
        _pad_token = token_info["input"]["TOK"]["TOK_PAD"]
        _stop_token = token_info["input"]["TOK"]["TOK_STOP"]
        _start_token = token_info["input"]["TOK"]["TOK_START"]

        # Extract valid tokens (skip special tokens)
        valid_tokens = tokens[
            (tokens != _pad_token) & (tokens != _start_token) & (tokens != _stop_token)
        ]

        if len(valid_tokens) == 0:
            return False

        # Convert to amino acid sequence using alphabet from token_info
        if "alphabet" in token_info and token_info["alphabet"] is not None:
            alphabet = token_info["alphabet"]
            alphabet = [
                item.decode("utf-8") if isinstance(item, bytes) else item
                for item in alphabet
            ]
            sequence = "".join([alphabet[int(token)] for token in valid_tokens])
        else:
            return False

        # Standard amino acids that should appear (excluding markers 1, 2, 3)
        standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        markers = {"1", "2", "3"}

        # Rule 1: Must start with '1' or '3'
        if not (sequence.startswith("1") or sequence.startswith("3")):
            return False

        count1 = sequence.count("1")
        count2 = sequence.count("2")
        count3 = sequence.count("3")

        # Rule 2, 3, 4: Check for presence and exact count of each marker
        if not (count1 == 1 and count2 == 1 and count3 == 1):
            return False

        # Rule 7: Check for allowed characters
        for char in sequence:
            if char not in markers and char not in standard_amino_acids:
                return False

        # Rules 5 and 6: Check marker order
        try:
            idx1 = sequence.index("1")
            idx2 = sequence.index("2")
            idx3 = sequence.index("3")

            if not (idx1 < idx2 and idx3 < idx2):
                return False
        except ValueError:
            return False

        return True

    except Exception as e:
        print(f"Error validating sequence characters: {e}")
        return False


def get_reward_function_registry():
    """
    Dynamically collect all reward functions from this module and custom_rewards.

    A reward function is identified by:
    1. Starting with "compute_"
    2. Being callable
    3. Having the signature: compute_*(tokens, token_info, device, **kwargs)

    Collects from:
    - idr_plm.nn.transformer.rewards (built-in functions)
    - custom_rewards (user-defined functions, if available)

    Returns:
        dict: Mapping of function names to function objects
    """
    registry = {}

    # Add built-in reward functions from this module
    current_module = sys.modules[__name__]

    for name in dir(current_module):
        if name.startswith("compute_") and callable(getattr(current_module, name)):
            func = getattr(current_module, name)
            registry[name] = func

    # Try to import and add custom reward functions
    try:
        import custom_rewards as custom_module

        for name in dir(custom_module):
            if name.startswith("compute_") and callable(getattr(custom_module, name)):
                func = getattr(custom_module, name)
                # Avoid overwriting built-in functions with custom ones
                if name not in registry:
                    registry[name] = func
                else:
                    print(
                        f"Warning: Custom reward function '{name}' shadows built-in function. Using built-in version."
                    )
    except ImportError:
        # custom_rewards module not found, which is fine - just use built-in functions
        pass

    return registry
