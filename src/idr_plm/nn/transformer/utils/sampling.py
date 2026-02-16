import time
import torch
from torch.nn.utils.rnn import pad_sequence


def forward_autoregressive_prompted(transformer_model, batch, token_sampler):
    """This method performs autoregressive sampling of SMILES strings or
    other tokenizer representations, but with a provided prompt.

    Notation for this function:
    N: batch size
    T: sequence length
    E: embedding dimension

    The expensive part of this operation is ensuring all sequences are the same length
    going into and coming out of the transformer. Consequently, this requires indexing based on
    the sum of sequence ids.
    """

    # Here, smi_batch is a list of tensors of differing lengths
    structural_batch, smi_batch, seq_id_batch = batch
    assert isinstance(smi_batch, list), "smi_batch should be a list of tensors"
    assert len(smi_batch) == seq_id_batch.shape[0] == structural_batch.shape[0]

    _pad_token = transformer_model.token_info["input"]["TOK"]["TOK_PAD"]
    _stop_token = transformer_model.token_info["input"]["TOK"]["TOK_STOP"]
    _start_token = transformer_model.token_info["input"]["TOK"]["TOK_START"]

    # Check if the first element of each smi_batch is _start_token, and if not, prepend the start token
    for i, smi_seq in enumerate(smi_batch):
        if len(smi_seq) == 0 or smi_seq[0] != _start_token:
            start_token_tensor = torch.tensor(
                [_start_token], device=smi_seq.device, dtype=smi_seq.dtype
            )
            smi_batch[i] = torch.cat([start_token_tensor, smi_seq], dim=0)

    # Construct correct input tensor and sequence id tensor
    working_smi_batch = pad_sequence(
        smi_batch, batch_first=True, padding_value=_pad_token
    )
    # Pad the batch with some placeholder padding tokens that will get populated. Pad by 50 tokens
    #   at a time
    working_smi_batch = torch.nn.functional.pad(
        working_smi_batch, (0, 50), mode="constant", value=_pad_token
    )  # (N, T + 1)
    working_seq_id_batch = (working_smi_batch != _pad_token).long()  # (N, T)
    working_structural_batch = (
        torch.ones(working_smi_batch.shape[0], 1) * _pad_token
    ).to(working_smi_batch.device)  # (N, 1)
    working_token_probs = torch.tensor([]).to(working_smi_batch.device)  # (N, 1)

    completed_smiles_structures = []
    completed_token_probs = []
    # FH: to keep track of the indices as they are completed
    completed_indices = []
    all_indices = torch.arange(working_smi_batch.shape[0]).to(
        working_smi_batch.device
    )  # (N,)

    # The ideal batch size assuming every SMILES is decoded properly. This value can
    # be updated if the effective batch size is reduced due to inproperly generated
    # padding tokens
    target_batch_size = working_smi_batch.shape[0]

    # Inference loop
    iteration = 0
    max_iterations = 1024  # See transformer_model.token_limit below too
    print()
    while len(completed_smiles_structures) < target_batch_size:
        iteration += 1
        transformer_model._inference_iteration = iteration

        if iteration % 10 == 0:
            print(
                f"Iteration {iteration}: {len(completed_smiles_structures)}/{target_batch_size} completed, batch_size={working_smi_batch.shape[0]}"
            )
        if iteration > max_iterations:
            print(f"WARNING: Reached max iterations ({max_iterations})")
            break
        # print(working_smi_batch)
        if transformer_model.training_args["training_mode"] == "era_online":
            logits = transformer_model.reference_model(
                working_smi_batch,
                working_structural_batch,
                working_seq_id_batch,
                batch_access_indices=all_indices,
            )  # (N, T, E)
        else:
            logits = transformer_model.model(
                working_smi_batch,
                working_structural_batch,
                working_seq_id_batch,
                batch_access_indices=all_indices,
            )  # (N, T, E)

        indexed_positions = working_seq_id_batch.sum(-1)  # (N,)
        # FH: Subtract 1 from the indexed position here because we want to last non-padding token.
        # The positions in indexed positions give the first padding token
        next_pos = logits[torch.arange(logits.shape[0]), indexed_positions - 1, :]
        # char_probs = torch.nn.functional.softmax(next_pos, dim=-1)  # (N, E)
        # tokens = torch.multinomial(char_probs, 1)  # (N, 1)
        # selected_probs = torch.gather(char_probs, 1, tokens)  # (N, 1)
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
        working_smi_batch[torch.arange(logits.shape[0]), indexed_positions] = (
            tokens_to_insert
        )
        working_token_probs = torch.cat((working_token_probs, selected_probs), dim=1)

        # Filter out by the stop token
        stop_token_mask = tokens.squeeze(-1) == _stop_token

        completed_smiles_structures.extend([*working_smi_batch[stop_token_mask]])
        completed_token_probs.extend([*working_token_probs[stop_token_mask]])
        completed_indices.extend([*all_indices[stop_token_mask]])

        working_smi_batch = working_smi_batch[~stop_token_mask]
        working_token_probs = working_token_probs[~stop_token_mask]
        working_structural_batch = working_structural_batch[~stop_token_mask]
        all_indices = all_indices[~stop_token_mask]
        tokens = tokens[~stop_token_mask]

        # Filter out by any sampled padding tokens
        ctrl_token_mask = tokens.squeeze(-1) >= _pad_token
        working_smi_batch = working_smi_batch[~ctrl_token_mask]
        working_token_probs = working_token_probs[~ctrl_token_mask]
        working_structural_batch = working_structural_batch[~ctrl_token_mask]
        all_indices = all_indices[~ctrl_token_mask]
        # Modify the target batch size in such cases
        target_batch_size = target_batch_size - ctrl_token_mask.sum().item()

        # Check if we have to expand the size of working_smi_batch
        last_tokens = working_smi_batch[:, -1]
        if (last_tokens != _pad_token).any():
            working_smi_batch = torch.nn.functional.pad(
                working_smi_batch, (0, 50), mode="constant", value=_pad_token
            )
        working_seq_id_batch = (working_smi_batch != _pad_token).long()  # (N, T)

    assert (
        len(completed_smiles_structures)
        == len(completed_token_probs)
        == len(completed_indices)
        == target_batch_size
    ), "Length mismatch between completed quantities"
    # FH: Note that because this is based on the original batch size, it is possible for there
    # to be None values in either the completed_smiles_structures or completed_token_probs. We
    # retain them to ensure that the indices are mapping into a list of the correct size.
    original_batch_size = max(completed_indices) + 1
    reordered_smiles_structures = [None] * original_batch_size
    reordered_token_probs = [None] * original_batch_size
    for i, idx in enumerate(completed_indices):
        reordered_smiles_structures[idx] = (
            completed_smiles_structures[i].detach().cpu().numpy()
            if completed_smiles_structures[i] is not None
            else None
        )
        reordered_token_probs[idx] = (
            completed_token_probs[i].detach().cpu().numpy()
            if completed_token_probs[i] is not None
            else None
        )

    # batch_time = time.time() - start_time
    # print(f"forward_autoregressive_prompted batch (size={original_batch_size}) took {batch_time:.3f}s")

    return reordered_smiles_structures, reordered_token_probs


def forward_autoregressive(transformer_model, batch, token_sampler):
    """Note that here, smi_batch should all be start tokens

    Notation for this function:

    N: batch size
    T: sequence length
    E: embedding dimension

    """
    structural_batch, smi_batch, seq_id_batch = batch
    assert structural_batch.shape[0] == smi_batch.shape[0] == seq_id_batch.shape[0]
    device = structural_batch.device
    batch_size = structural_batch.shape[0]
    # For keeping track and only working on incomplete structures
    index_mapping = torch.arange(batch_size, device=device)
    completed_structures = [None] * batch_size
    # Save the sum of log probabilities as a score for each sequence
    completed_token_probs = [None] * batch_size
    all_structures_completed = False

    # Three quantities to keep track of and grow as we go
    working_smi_batch = smi_batch.clone()  # (N, 1)
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
            working_smi_batch.shape,
            working_structural_batch.shape,
            working_seq_id_batch.shape,
        )
        logits = transformer_model.model(
            working_smi_batch, working_structural_batch, working_seq_id_batch
        )  # (N, T, E)
        next_pos = logits[:, -1, :]  # (N, E)
        # char_probs = torch.nn.functional.softmax(next_pos, dim=-1)  # (N, E)
        # tokens = torch.multinomial(char_probs, 1)  # (N, 1)
        # selected_probs = torch.gather(char_probs, 1, tokens)  # (N, 1)
        tokens, selected_probs = token_sampler(
            next_pos
        )  # (N, 1), sampler directly callable

        concatenated_results = torch.cat((working_smi_batch, tokens), dim=-1)
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

        working_smi_batch = concatenated_results[~stop_token_mask]
        working_token_probs = concatenated_probs[~stop_token_mask]
        index_mapping = index_mapping[~stop_token_mask]
        # Also update the sequence ID and structural tokens
        working_seq_id_batch = concatenated_seq_id[~stop_token_mask]
        working_structural_batch = concatenated_structural[~stop_token_mask]

        if (
            working_smi_batch.shape[-1] > transformer_model.token_limit
        ):  # Fix this hardcoding for the token limit
            # NOTE: transformer_model.token_limit is now set in training config with:
            # "++training.lightning_model_args.sampler_args.token_limit=1000"\

            working_smi_batch = torch.cat(
                (
                    working_smi_batch,
                    torch.tensor([stop_token] * working_smi_batch.shape[0])
                    .reshape(-1, 1)
                    .to(device),
                ),
                dim=-1,
            )
            working_token_probs = torch.cat(
                (
                    working_token_probs,
                    torch.tensor([0.0] * working_smi_batch.shape[0])
                    .reshape(-1, 1)
                    .to(device),
                ),
                dim=-1,
            )
            for j, idx in enumerate(index_mapping):
                completed_structures[idx] = working_smi_batch[j].detach().cpu().numpy()
                completed_token_probs[idx] = (
                    working_token_probs[j].detach().cpu().numpy()
                )

            all_structures_completed = True

        if len(working_smi_batch) == 0:
            all_structures_completed = True
        # import pdb; pdb.set_trace()

    return completed_structures, completed_token_probs


def sample_components_from_autoregressive_transformer(
    transformer_model,
    structural_tokens,
    smiles_tokens,
    sequence_id,
    token_sampler,
    inference_batch_size=128,
    use_input_smiles=False,
):
    """
    Samples from the transformer autoregressively, i.e. next token prediction left to right

    Have to be careful here because the sequence ID determines the autoregressive mask of the model
    used during MHA
    """
    # import pdb; pdb.set_trace()
    transformer_model.model.eval()
    device = transformer_model.device
    assert not transformer_model.model.training
    if isinstance(smiles_tokens, torch.Tensor):
        num_batches = (
            smiles_tokens.shape[0] + inference_batch_size - 1
        ) // inference_batch_size
    elif isinstance(smiles_tokens, list):
        num_batches = (
            len(smiles_tokens) + inference_batch_size - 1
        ) // inference_batch_size
    else:
        raise ValueError("Unrecognized data format for smiles tokens")
    all_sampled_tokens = []
    all_token_probs = []
    for batch in range(num_batches):
        batch_start_time = time.time()
        print(batch, num_batches)
        structural_tokens_batch = structural_tokens[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        smiles_tokens_batch = smiles_tokens[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        sequence_id_batch = sequence_id[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        # FH: Device cast here only at run time when the batch is created
        structural_tokens_batch = structural_tokens_batch.to(device)
        sequence_id_batch = sequence_id_batch.to(device)
        if isinstance(smiles_tokens_batch, torch.Tensor):
            smiles_tokens_batch = smiles_tokens_batch.to(device)
        elif isinstance(smiles_tokens_batch, list):
            smiles_tokens_batch = [
                x.to(device) if isinstance(x, torch.Tensor) else x
                for x in smiles_tokens_batch
            ]
        if not use_input_smiles:
            print("Generating sequences unprompted")
            tokens, probs = forward_autoregressive(
                transformer_model,
                (structural_tokens_batch, smiles_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        else:
            print("Generating sequences prompted")
            tokens, probs = forward_autoregressive_prompted(
                transformer_model,
                (structural_tokens_batch, smiles_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        all_sampled_tokens.append(tokens)
        all_token_probs.append(probs)
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch}/{num_batches} completed in {batch_time:.2f} seconds")

    return all_sampled_tokens, all_token_probs
