import torch
import torch.nn as nn


def choose_positions(logits, mask, criterion, top_k):
    """
    Choosing either based on maximum entropy or maximum logit

    logits: (seq_len, vocab_size) Logits over the tokens
    mask: (seq_len) Indicates which positions are masked or not
    criterion: How to choose positions, either:
        entropy: Chooses position with minimum entropy after softmax
        max_logit: Chooses position with maximum logit
    top_k: The top k positions to take for the positions to unmask
        e.g. if k = 5, the top 5 lowest entropy or highest logits are retained as positions
    """
    mask = mask.bool()  # For easier selection
    masked_position_logits = logits[mask]  # (m, vocab_size)
    if criterion == "entropy":
        p = nn.functional.softmax(masked_position_logits, dim=-1)
        entropies = (p * torch.log(p)).sum(-1)
        minimum_pos = torch.argsort(entropies)[:top_k]
        return minimum_pos
    elif criterion == "max_logit":
        maximums = torch.max(masked_position_logits, dim=-1)
        maximum_pos = torch.argsort(maximums)[-top_k:]
        return maximum_pos


def generate_output(model, batch, n_decode, criterion, top_k):
    """
    model: Trained model for inference
    batch: Combination of prompts and masks
    n_decode: Number to decode
    criterion: How to select the next positions
    top_k: The top k positions to take for the positions to unmask
    """

    # prompts: (batch_size, seq_len)
    # masks: (batch_size, seq_len)
    prompts, masks = batch

    for _ in range(n_decode):
        logits = model(prompts)  # (batch_size, seq_len, vocab_size)

        for b_idx in range(logits.shape[0]):
            curr_logits = logits[b_idx]  # (seq_len, vocab_size)
            positions = choose_positions(
                curr_logits, masks[b_idx], criterion, top_k
            )  # Choose a position

            mask_token_probs = nn.functional.softmax(curr_logits, dim=-1)
            mask_token_probs = mask_token_probs[positions]
            # Sample one token for now
            tokens = torch.multinomial(mask_token_probs, 1)
            # Put tokens back into the correct place (update prompt)
            prompts[b_idx][positions] = tokens

    # Once prompts are complete, they are the completions
    return prompts


# TODO:
# 1. Add temperature to softmax
# 2. Figure out batching
# 3. Reindex positions
# 4. Update loss mask and mask_token_probs
