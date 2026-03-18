import time
import torch
from torch.nn.utils.rnn import pad_sequence
from idr_plm.nn.transformer.nn import GeometricMolTransformer
from idr_plm.utils.sampler import TokenSampler
from idr_plm.nn.transformer import scores


def forward_autoregressive_prompted(transformer_model, batch, token_sampler):
    """This method performs autoregressive sampling of residue sequences or
    other tokenizer representations, but with a provided prompt.

    Notation for this function:
    N: batch size
    T: sequence length
    E: embedding dimension

    The expensive part of this operation is ensuring all sequences are the same length
    going into and coming out of the transformer. Consequently, this requires indexing based on
    the sum of sequence ids.
    """

    # Here, res_batch is a list of tensors of differing lengths
    structural_batch, res_batch, seq_id_batch = batch
    assert isinstance(res_batch, list), "res_batch should be a list of tensors"
    assert len(res_batch) == seq_id_batch.shape[0] == structural_batch.shape[0]

    _pad_token = transformer_model.token_info["input"]["TOK"]["TOK_PAD"]
    _stop_token = transformer_model.token_info["input"]["TOK"]["TOK_STOP"]
    _start_token = transformer_model.token_info["input"]["TOK"]["TOK_START"]

    # Check if the first element of each res_batch is _start_token, and if not, prepend the start token
    for i, res_seq in enumerate(res_batch):
        if len(res_seq) == 0 or res_seq[0] != _start_token:
            start_token_tensor = torch.tensor(
                [_start_token], device=res_seq.device, dtype=res_seq.dtype
            )
            res_batch[i] = torch.cat([start_token_tensor, res_seq], dim=0)

    # Construct correct input tensor and sequence id tensor
    working_res_batch = pad_sequence(
        res_batch, batch_first=True, padding_value=_pad_token
    )
    # Pad the batch with some placeholder padding tokens that will get populated. Pad by 50 tokens
    #   at a time
    working_res_batch = torch.nn.functional.pad(
        working_res_batch, (0, 50), mode="constant", value=_pad_token
    )  # (N, T + 1)
    working_seq_id_batch = (working_res_batch != _pad_token).long()  # (N, T)
    working_structural_batch = (
        torch.ones(working_res_batch.shape[0], 1) * _pad_token
    ).to(working_res_batch.device)  # (N, 1)
    working_token_probs = torch.tensor([]).to(working_res_batch.device)  # (N, 1)

    completed_residues_structures = []
    completed_token_probs = []
    # FH: to keep track of the indices as they are completed
    completed_indices = []
    all_indices = torch.arange(working_res_batch.shape[0]).to(
        working_res_batch.device
    )  # (N,)

    # The ideal batch size assuming every residue sequence is decoded properly. This value can
    # be updated if the effective batch size is reduced due to improperly generated
    # padding tokens
    target_batch_size = working_res_batch.shape[0]

    # Inference loop
    iteration = 0
    max_iterations = 1024  # See transformer_model.token_limit below too
    print()
    while len(completed_residues_structures) < target_batch_size:
        iteration += 1
        transformer_model._inference_iteration = iteration

        if iteration % 10 == 0:
            print(
                f"Iteration {iteration}: {len(completed_residues_structures)}/{target_batch_size} completed, batch_size={working_res_batch.shape[0]}"
            )
        if iteration > max_iterations:
            print(f"WARNING: Reached max iterations ({max_iterations})")
            break
        if transformer_model.training_args["training_mode"] == "era_online":
            logits = transformer_model.reference_model(
                working_res_batch,
                working_structural_batch,
                working_seq_id_batch,
                batch_access_indices=all_indices,
            )  # (N, T, E)
        else:
            logits = transformer_model.model(
                working_res_batch,
                working_structural_batch,
                working_seq_id_batch,
                batch_access_indices=all_indices,
            )  # (N, T, E)

        indexed_positions = working_seq_id_batch.sum(-1)  # (N,)
        # FH: Subtract 1 from the indexed position here because we want to last non-padding token.
        # The positions in indexed positions give the first padding token
        next_pos = logits[torch.arange(logits.shape[0]), indexed_positions - 1, :]
        tokens, selected_probs = token_sampler(next_pos)

        # Check if next_pos is outside of transformer_model.token_limit
        # If it is, then append a stop token
        # NOTE: transformer_model.token_limit is now set in training config with:
        # "++training.lightning_model_args.sampler_args.token_limit=1000"\
        position_exceeds_limit = indexed_positions > transformer_model.token_limit
        tokens_to_insert = tokens.squeeze(-1).clone()
        tokens_to_insert[position_exceeds_limit] = _stop_token

        # FH: Here, indexed_positions is used as is because we want to update the first padding token
        # as the sampled non-padding token
        working_res_batch[torch.arange(logits.shape[0]), indexed_positions] = (
            tokens_to_insert
        )
        working_token_probs = torch.cat((working_token_probs, selected_probs), dim=1)

        # Filter out by the stop token
        stop_token_mask = tokens.squeeze(-1) == _stop_token

        completed_residues_structures.extend([*working_res_batch[stop_token_mask]])
        completed_token_probs.extend([*working_token_probs[stop_token_mask]])
        completed_indices.extend([*all_indices[stop_token_mask]])

        working_res_batch = working_res_batch[~stop_token_mask]
        working_token_probs = working_token_probs[~stop_token_mask]
        working_structural_batch = working_structural_batch[~stop_token_mask]
        all_indices = all_indices[~stop_token_mask]
        tokens = tokens[~stop_token_mask]

        # Filter out by any sampled padding tokens
        ctrl_token_mask = tokens.squeeze(-1) >= _pad_token
        working_res_batch = working_res_batch[~ctrl_token_mask]
        working_token_probs = working_token_probs[~ctrl_token_mask]
        working_structural_batch = working_structural_batch[~ctrl_token_mask]
        all_indices = all_indices[~ctrl_token_mask]
        # Modify the target batch size in such cases
        target_batch_size = target_batch_size - ctrl_token_mask.sum().item()

        # Check if we have to expand the size of working_res_batch
        last_tokens = working_res_batch[:, -1]
        if (last_tokens != _pad_token).any():
            working_res_batch = torch.nn.functional.pad(
                working_res_batch, (0, 50), mode="constant", value=_pad_token
            )
        working_seq_id_batch = (working_res_batch != _pad_token).long()  # (N, T)

    assert (
        len(completed_residues_structures)
        == len(completed_token_probs)
        == len(completed_indices)
        == target_batch_size
    ), "Length mismatch between completed quantities"
    # FH: Note that because this is based on the original batch size, it is possible for there
    # to be None values in either the completed_residues_structures or completed_token_probs. We
    # retain them to ensure that the indices are mapping into a list of the correct size.
    original_batch_size = max(completed_indices) + 1
    reordered_residues_structures = [None] * original_batch_size
    reordered_token_probs = [None] * original_batch_size
    for i, idx in enumerate(completed_indices):
        reordered_residues_structures[idx] = (
            completed_residues_structures[i].detach().cpu().numpy()
            if completed_residues_structures[i] is not None
            else None
        )
        reordered_token_probs[idx] = (
            completed_token_probs[i].detach().cpu().numpy()
            if completed_token_probs[i] is not None
            else None
        )

    return reordered_residues_structures, reordered_token_probs


def forward_autoregressive(transformer_model, batch, token_sampler):
    """Note that here, res_batch should all be start tokens

    Notation for this function:

    N: batch size
    T: sequence length
    E: embedding dimension

    """
    structural_batch, res_batch, seq_id_batch = batch
    assert structural_batch.shape[0] == res_batch.shape[0] == seq_id_batch.shape[0]
    device = structural_batch.device
    batch_size = structural_batch.shape[0]
    # For keeping track and only working on incomplete structures
    index_mapping = torch.arange(batch_size, device=device)
    completed_structures = [None] * batch_size
    # Save the sum of log probabilities as a score for each sequence
    completed_token_probs = [None] * batch_size
    all_structures_completed = False

    # Three quantities to keep track of and grow as we go
    working_res_batch = res_batch.clone()  # (N, 1)
    working_seq_id_batch = seq_id_batch.clone()  # (N, 1)
    working_structural_batch = structural_batch.clone()  # (N, 1)
    working_token_probs = torch.tensor([]).to(device)

    # mask_token = transformer_model.token_info['input']['TOK']['TOK_MASK']
    pad_token = transformer_model.token_info["input"]["TOK"]["TOK_PAD"]
    stop_token = transformer_model.token_info["input"]["TOK"]["TOK_STOP"]
    # start_token = transformer_model.token_info['input']['TOK']['TOK_START']

    while not all_structures_completed:
        print(
            "beginning of loop",
            working_res_batch.shape,
            working_structural_batch.shape,
            working_seq_id_batch.shape,
        )
        logits = transformer_model.model(
            working_res_batch, working_structural_batch, working_seq_id_batch
        )  # (N, T, E)
        next_pos = logits[:, -1, :]  # (N, E)
        tokens, selected_probs = token_sampler(
            next_pos
        )  # (N, 1), sampler directly callable

        concatenated_results = torch.cat((working_res_batch, tokens), dim=-1)
        concatenated_probs = torch.cat((working_token_probs, selected_probs), dim=1)
        # Sequence id here is binary so appending on more 1's is good enough
        concatenated_seq_id = torch.cat(
            (
                working_seq_id_batch,
                torch.ones((working_seq_id_batch.shape[0], 1), device=device).long(),
            ),
            dim=1,
        )
        concatenated_structural = torch.cat(
            (
                working_structural_batch,
                torch.ones((working_structural_batch.shape[0], 1), device=device).long()
                * pad_token,
            ),
            dim=1,
        )

        stop_token_mask = concatenated_results[:, -1] == stop_token
        comp_structs = concatenated_results[stop_token_mask]
        comp_probs = concatenated_probs[stop_token_mask]
        comp_inds = index_mapping[stop_token_mask]

        for i, icomp in enumerate(comp_inds):
            completed_structures[icomp] = comp_structs[i].detach().cpu().numpy()
            completed_token_probs[icomp] = comp_probs[i].detach().cpu().numpy()

        working_res_batch = concatenated_results[~stop_token_mask]
        working_token_probs = concatenated_probs[~stop_token_mask]
        index_mapping = index_mapping[~stop_token_mask]
        # Also update the sequence ID and structural tokens
        working_seq_id_batch = concatenated_seq_id[~stop_token_mask]
        working_structural_batch = concatenated_structural[~stop_token_mask]

        if (
            working_res_batch.shape[-1] > transformer_model.token_limit
        ):  # Fix this hardcoding for the token limit
            # NOTE: transformer_model.token_limit is now set in training config with:
            # "++training.lightning_model_args.sampler_args.token_limit=1000"\

            working_res_batch = torch.cat(
                (
                    working_res_batch,
                    torch.tensor([stop_token] * working_res_batch.shape[0])
                    .reshape(-1, 1)
                    .to(device),
                ),
                dim=-1,
            )
            working_token_probs = torch.cat(
                (
                    working_token_probs,
                    torch.tensor([0.0] * working_res_batch.shape[0])
                    .reshape(-1, 1)
                    .to(device),
                ),
                dim=-1,
            )
            for j, idx in enumerate(index_mapping):
                completed_structures[idx] = working_res_batch[j].detach().cpu().numpy()
                completed_token_probs[idx] = (
                    working_token_probs[j].detach().cpu().numpy()
                )

            all_structures_completed = True

        if len(working_res_batch) == 0:
            all_structures_completed = True

    return completed_structures, completed_token_probs


def sample_components_from_autoregressive_transformer(
    transformer_model,
    structural_tokens,
    res_tokens,
    sequence_id,
    token_sampler,
    inference_batch_size=128,
    use_input_residues=False,
):
    """
    Samples from the transformer autoregressively, i.e. next token prediction left to right

    Have to be careful here because the sequence ID determines the autoregressive mask of the model
    used during MHA
    """
    transformer_model.model.eval()
    device = transformer_model.device
    assert not transformer_model.model.training
    if isinstance(res_tokens, torch.Tensor):
        num_batches = (
            res_tokens.shape[0] + inference_batch_size - 1
        ) // inference_batch_size
    elif isinstance(res_tokens, list):
        num_batches = (
            len(res_tokens) + inference_batch_size - 1
        ) // inference_batch_size
    else:
        raise ValueError("Unrecognized data format for residue tokens")
    all_sampled_tokens = []
    all_token_probs = []
    for batch in range(num_batches):
        batch_start_time = time.time()
        print(batch, num_batches)
        structural_tokens_batch = structural_tokens[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        res_tokens_batch = res_tokens[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        sequence_id_batch = sequence_id[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        # FH: Device cast here only at run time when the batch is created
        structural_tokens_batch = structural_tokens_batch.to(device)
        sequence_id_batch = sequence_id_batch.to(device)
        if isinstance(res_tokens_batch, torch.Tensor):
            res_tokens_batch = res_tokens_batch.to(device)
        elif isinstance(res_tokens_batch, list):
            res_tokens_batch = [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in res_tokens_batch
            ]
        if not use_input_residues:
            print("Generating sequences unprompted")
            tokens, probs = forward_autoregressive(
                transformer_model,
                (structural_tokens_batch, res_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        else:
            print("Generating sequences prompted")
            tokens, probs = forward_autoregressive_prompted(
                transformer_model,
                (structural_tokens_batch, res_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        all_sampled_tokens.append(tokens)
        all_token_probs.append(probs)
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch}/{num_batches} completed in {batch_time:.2f} seconds")

    return all_sampled_tokens, all_token_probs


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
    res_batch_list = [repeated_tokens[i] for i in range(total_sequences)]

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
            (structural_tokens, res_batch_list, repeated_seq_id),
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
