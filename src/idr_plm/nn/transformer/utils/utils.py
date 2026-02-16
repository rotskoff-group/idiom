import torch
from idr_plm.nn.transformer.nn import GeometricMolTransformer
from idr_plm.utils.sampler import TokenSampler
from idr_plm.nn.transformer import scores
from idr_plm.nn.transformer.utils.sampling import forward_autoregressive_prompted


def compute_policy_logps(model, tokens, structure, masks):
    """Calculate per-token logps under a provided model

    Args:
        model: The model to compute logps with
        tokens: Token indices
        structure: Structural information
        masks: Attention masks

    Returns:
        policy_logps: Per-token log probabilities (1 shorter than len(tokens) because start token doesn't have a logp)
    """
    if isinstance(model, GeometricMolTransformer):
        policy_logits = model(tokens, structure, masks)

    policy_logps = policy_logits.log_softmax(dim=-1)
    policy_logps = torch.gather(
        policy_logps[:, :-1, :], dim=-1, index=tokens[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    # policy_logps is going to be 1 shorter than len(tokens) because start token doesn't have a logp
    return policy_logps


def generate_sequences_online(lightning_model, tokens, masks, group_size):
    """Generate group_size completions for each prompt using forward_autoregressive_prompted

    Args:
        lightning_model: The LightningModel instance
        tokens: (batch_size, seq_len) - prompt tokens
        masks: (batch_size, seq_len) - attention masks for prompts
        group_size: int - number of completions to generate per prompt

    Returns:
        tuple: (all_generated_sequences, final_num_invalid)
            - all_generated_sequences: list of generated sequences with completions for GRPO training
            - final_num_invalid: number of invalid sequences
    """

    _pad_token = lightning_model.token_info["input"]["TOK"]["TOK_PAD"]
    _stop_token = lightning_model.token_info["input"]["TOK"]["TOK_STOP"]
    _start_token = lightning_model.token_info["input"]["TOK"]["TOK_START"]

    batch_size = tokens.shape[0]
    all_generated_sequences = []

    # Generate group_size completions for all prompts in parallel

    # Repeat each prompt group_size times: (batch_size, seq_len) -> (batch_size * group_size, seq_len)
    # repeat_interleave keeps the duplicated prompts together
    repeated_tokens = tokens.repeat_interleave(
        group_size, dim=0
    )  # (batch_size * group_size, seq_len)
    repeated_seq_id = masks.repeat_interleave(
        group_size, dim=0
    )  # (batch_size * group_size, seq_len)

    # Create dummy structural tokens for all sequences
    total_sequences = batch_size * group_size
    if isinstance(lightning_model.model, GeometricMolTransformer):
        structural_tokens = torch.zeros(
            (total_sequences, 1), dtype=torch.long, device=lightning_model.device
        )
    else:
        structural_tokens = torch.zeros(
            (total_sequences, 1), dtype=torch.long, device=lightning_model.device
        )

    # Convert tokens to list format expected by forward_autoregressive_prompted
    smi_batch_list = [repeated_tokens[i] for i in range(total_sequences)]

    # Generate completions for all prompts at once
    with torch.no_grad():
        # Create a TokenSampler for this generation if not available
        if (
            hasattr(lightning_model, "token_sampler")
            and lightning_model.token_sampler is not None
        ):
            sampler = lightning_model.token_sampler
        else:
            sampler = TokenSampler(
                method=lightning_model.sampling_method,
                sample_val=lightning_model.sample_val,
                temperature=lightning_model.sampling_temperature,
            )

        # Generate completions using forward_autoregressive_prompted
        completed_sequences, completed_probs = forward_autoregressive_prompted(
            lightning_model,
            (structural_tokens, smi_batch_list, repeated_seq_id),
            token_sampler=sampler,
        )

    # Process all completed sequences and organize by prompt
    for i in range(total_sequences):
        if completed_sequences[i] is not None:
            # Calculate which prompt this sequence belongs to
            prompt_idx = i // group_size

            seq = torch.tensor(
                completed_sequences[i], device=lightning_model.device
            )  # This includes start, stop, and pad tokens
            # Non-pad token mask:
            seq_ids = (seq != _pad_token).long()
            prob = completed_probs[i] if completed_probs[i] is not None else None

            # Find the actual prompt length (excluding padding)
            prompt_tokens = repeated_tokens[
                i
            ]  # Here, this doesn't include start or stop tokens
            prompt_length = (prompt_tokens != _pad_token).sum().item()

            # Create response mask: 1 for generated tokens, 0 for prompt and pad tokens
            response_mask = torch.zeros_like(seq)

            # Mark generated tokens (everything after the prompt) as 1
            if len(seq) > prompt_length:
                response_mask[prompt_length + 1 :] = 1
                # Above, prompt_length + 1 because prompt_length doesn't include start token

            # Ensure pad tokens remain 0
            response_mask[seq == _pad_token] = 0

            all_generated_sequences.append(
                {
                    "sequence": seq,
                    "response_mask": response_mask,
                    "seq_ids": seq_ids,
                    "probability": prob,
                    "prompt_idx": prompt_idx,
                }
            )
        # This contains (batch_size * group_size) generated sequences and probs

    # Initialize validity checking for generated sequences (when too far off-policy, there may be invalid sequences)
    if not hasattr(lightning_model, "check_sequence_validity"):
        lightning_model.check_sequence_validity = scores.valid_sequence_characters

    if not hasattr(lightning_model, "log_invalid_sequences"):
        lightning_model.log_invalid_sequences = True

    if not hasattr(lightning_model, "_regeneration_depth"):
        lightning_model._regeneration_depth = 0
        lightning_model._max_regeneration_depth = 5

    # Check for invalid sequences
    invalid_indices = []
    for i, seq_data in enumerate(all_generated_sequences):
        if not lightning_model.check_sequence_validity(
            seq_data["sequence"], lightning_model.token_info, lightning_model.device
        ):
            invalid_indices.append(i)

    # Track the final number of invalid sequences
    final_num_invalid = len(invalid_indices)

    # Recursive regeneration if invalid sequences found
    if len(invalid_indices) > 0:
        if (
            lightning_model._regeneration_depth
            < lightning_model._max_regeneration_depth
        ):
            if lightning_model.log_invalid_sequences:
                print(
                    f"Step {lightning_model.global_step}: Found {len(invalid_indices)} invalid sequences, recursing (depth {lightning_model._regeneration_depth + 1}/{lightning_model._max_regeneration_depth})"
                )

            lightning_model._regeneration_depth += 1
            # Recursively call to regenerate all sequences
            sequences, num_invalid = generate_sequences_online(
                lightning_model, tokens, masks, group_size
            )
            return sequences, num_invalid
        else:
            if lightning_model.log_invalid_sequences:
                print(
                    f"Step {lightning_model.global_step}: {len(invalid_indices)} sequences remain invalid after max depth"
                )
            lightning_model._regeneration_depth = 0
            raise RuntimeError(
                f"Step {lightning_model.global_step}: Max regeneration depth ({lightning_model._max_regeneration_depth}) exceeded. "
                f"{len(invalid_indices)} sequences remain invalid. Training halted to prevent propagation of invalid sequences."
            )
    else:
        # All valid, reset depth counter
        lightning_model._regeneration_depth = 0

    return all_generated_sequences, final_num_invalid
