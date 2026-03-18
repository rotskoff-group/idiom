import time
import torch
from torch.nn.utils.rnn import pad_sequence

from idr_plm.nn.transformer import scores
from idr_plm.nn.transformer.utils.misc import compute_policy_logps
from idr_plm.nn.transformer.utils.sampling import generate_sequences_online


def _initialize_grpo_config(lightning_module):
    """Initialize GRPO config once on the LightningModule."""
    if getattr(lightning_module, "_grpo_config_initialized", False):
        return

    grpo_args = lightning_module.hparams.training_args.lightning_model_args

    lightning_module.group_size = torch.tensor(
        grpo_args.get("group_size", 4), device=lightning_module.device
    )
    lightning_module.epsilon_clip = torch.tensor(
        grpo_args.get("epsilon_clip", 0.2), device=lightning_module.device
    )
    # Number off-policy steps. Should keep this at 1
    lightning_module.mu_grpo = torch.tensor(
        grpo_args.get("mu_grpo", 1), device=lightning_module.device
    )
    # Magnitude of D_KL penalty
    lightning_module.beta_kl = torch.tensor(
        grpo_args.get("beta_kl", 0.1), device=lightning_module.device
    )

    # Quadratic reward shaping
    lightning_module.use_reward_shaping = grpo_args.get("use_reward_shaping", False)
    lightning_module.reward_target_value = torch.tensor(
        grpo_args.get("reward_target_value", 0.5), device=lightning_module.device
    )
    lightning_module.reward_scale = torch.tensor(
        grpo_args.get("reward_scale", 1.0), device=lightning_module.device
    )

    # Percent identity calculation sampling fraction. Just for diagnostics
    lightning_module.pid_sample_fraction = grpo_args.get("pid_sample_fraction", 0.25)

    # Advantage normalization flag
    lightning_module.normalize_advantage = grpo_args.get("normalize_advantage", True)

    # Entropy reward parameters
    lightning_module.use_target_entropy = grpo_args.get("use_target_entropy", False)
    lightning_module.target_entropy = torch.tensor(
        grpo_args.get("target_entropy", 2.75), device=lightning_module.device
    )
    lightning_module.entropy_reward_weight = torch.tensor(
        grpo_args.get("entropy_reward_weight", 1.0), device=lightning_module.device
    )
    lightning_module.entropy_reward_width = grpo_args.get("entropy_reward_width", 0.2)

    # Reward function selection
    lightning_module.reward_function_name = grpo_args.get(
        "reward_function_name", "compute_fraction_alanine"
    )

    # ProtGPS-specific parameters
    lightning_module.protgps_target_compartment = grpo_args.get(
        "protgps_target_compartment", "p-body"
    )
    lightning_module.protgps_parent_dir = grpo_args.get(
        "protgps_parent_dir", "/home/protgps"
    )
    lightning_module.protgps_aggregation = grpo_args.get("protgps_aggregation", "max")

    # Target length reward parameters
    lightning_module.use_target_length = grpo_args.get("use_target_length", False)
    lightning_module.target_length = torch.tensor(
        grpo_args.get("target_length", 100), device=lightning_module.device
    )
    lightning_module.length_reward_weight = torch.tensor(
        grpo_args.get("length_reward_weight", 1.0), device=lightning_module.device
    )
    # Controls the width of the length reward curve. Higher values allow more variability.
    # Default 1.0 is tight; increase to 2.0, 3.0 etc for wider tolerance around target length
    lightning_module.length_reward_width = grpo_args.get("length_reward_width", 0.1)

    # Map reward function name to actual function (loads from scores.py as well as custom_rewards.py in entrypoints)
    reward_function_registry = scores.get_reward_function_registry()

    if lightning_module.reward_function_name not in reward_function_registry:
        available_functions = ", ".join(reward_function_registry.keys())
        raise ValueError(
            f"Reward function '{lightning_module.reward_function_name}' not found. "
            f"Available functions: {available_functions}"
        )

    lightning_module.reward_function = reward_function_registry[
        lightning_module.reward_function_name
    ]
    lightning_module._grpo_config_initialized = True


def _compute_and_log_percent_identity(lightning_module, generated_sequences, prefix):
    # Calculate all-to-all percent identities using biopython
    percent_identity_start = time.time()

    # Just for logging:
    percent_identities = scores.calculate_percent_identities(
        generated_sequences,
        token_info=lightning_module.token_info,
        global_align=False,  # When False, normalize by min(len(seq1), len(seq2))
        sample_fraction=lightning_module.pid_sample_fraction,
    )

    percent_identity_time = time.time() - percent_identity_start
    print(
        f"Step {lightning_module.global_step}: Percent identity calculation took {percent_identity_time:.3f}s"
    )

    # Calculate and log percent IDs
    percent_identities_tensor = torch.tensor(
        percent_identities, device=lightning_module.device
    )
    mean_pid = percent_identities_tensor.mean().item()
    std_pid = percent_identities_tensor.std().item()

    print(
        f"Step {lightning_module.global_step}: Mean percent identity: {mean_pid:.2f}%, Std: {std_pid:.2f}%"
    )

    # Log to metrics
    pid_metrics = {
        f"{prefix}/mean_percent_identity": mean_pid,
        f"{prefix}/std_percent_identity": std_pid,
    }
    lightning_module.log_dict(
        pid_metrics,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
        prog_bar=True,
        logger=True,
    )


def _compute_and_log_rewards(
    lightning_module, generated_sequences, num_invalid_sequences, prefix
):
    # Compute rewards for generated sequences
    reward_start = time.time()
    rewards = []
    raw_rewards = []  # Store raw rewards before shaping
    length_rewards = []  # Store length rewards separately for logging
    entropy_rewards = []  # Store entropy rewards separately for logging
    entropies = []  # Store Shannon entropy values for logging

    for seq_data in generated_sequences:  # Reward calculation is serial here
        # Extract the sequence tensor
        sequence = seq_data["sequence"]  # tokens
        # Sequences are passed into score as tokens
        # Here, sequences will contain start_token, stop_token, and pad_tokens
        # Alphabet is contained in lightning_module.token_info

        # Call reward function with appropriate parameters
        if lightning_module.reward_function_name == "compute_protgps_score":
            raw_reward = lightning_module.reward_function(
                sequence,
                lightning_module.token_info,
                lightning_module.device,
                target_compartment=lightning_module.protgps_target_compartment,
                protgps_parent_dir=lightning_module.protgps_parent_dir,
                aggregation=lightning_module.protgps_aggregation,
            )
        else:
            raw_reward = lightning_module.reward_function(
                sequence, lightning_module.token_info, lightning_module.device
            )

        raw_rewards.append(raw_reward)

        # Apply reward shaping if enabled
        if lightning_module.use_reward_shaping:
            shaped_reward = scores.apply_quadratic_reward_shaping(
                raw_reward,
                lightning_module.reward_target_value,
                lightning_module.reward_scale,
            )
        else:
            shaped_reward = raw_reward

        # Compute length reward if enabled and add to total reward
        if lightning_module.use_target_length:
            length_reward = scores.compute_length_reward(
                sequence,
                lightning_module.token_info,
                lightning_module.device,
                target_length=lightning_module.target_length.item(),
                length_reward_width=lightning_module.length_reward_width,
            )
            length_rewards.append(length_reward)
            total_reward = (
                shaped_reward + lightning_module.length_reward_weight * length_reward
            )
        else:
            length_rewards.append(torch.tensor(0.0, device=lightning_module.device))
            total_reward = shaped_reward

        # Compute entropy reward if enabled and add to total reward
        # Also compute entropy for logging regardless
        entropy = scores.compute_sequence_entropy(
            sequence, lightning_module.token_info, lightning_module.device
        )
        entropies.append(entropy)
        if lightning_module.use_target_entropy:
            entropy_reward = scores.compute_entropy_reward(
                sequence,
                lightning_module.token_info,
                lightning_module.device,
                target_entropy=lightning_module.target_entropy.item(),
                entropy_reward_width=lightning_module.entropy_reward_width,
            )
            entropy_rewards.append(entropy_reward)
            total_reward = (
                total_reward + lightning_module.entropy_reward_weight * entropy_reward
            )
        else:
            entropy_rewards.append(torch.tensor(0.0, device=lightning_module.device))

        rewards.append(total_reward)

    # Convert rewards to tensor
    rewards_tensor = torch.stack(rewards)
    raw_rewards_tensor = torch.stack(raw_rewards)
    length_rewards_tensor = torch.stack(length_rewards)
    entropy_rewards_tensor = torch.stack(entropy_rewards)
    entropies_tensor = torch.stack(entropies)
    reward_time = time.time() - reward_start
    print(
        f"Step {lightning_module.global_step}: Reward computation took {reward_time:.3f}s"
    )

    # Calculate IDR lengths (excluding pad tokens)
    seq_lengths_tensor = scores.calculate_idr_length(
        generated_sequences, lightning_module.token_info, lightning_module.device
    )

    # Log reward statistics
    metrics = {
        f"{prefix}/mean_reward": rewards_tensor.mean(),
        f"{prefix}/std_reward": rewards_tensor.std(),
        f"{prefix}/mean_seq_length": seq_lengths_tensor.mean(),
        f"{prefix}/std_seq_length": seq_lengths_tensor.std(),
        f"{prefix}/mean_entropy": entropies_tensor.mean(),
        f"{prefix}/std_entropy": entropies_tensor.std(),
        f"{prefix}/num_invalid_sequences": num_invalid_sequences,
    }

    # Log raw reward statistics if reward shaping is enabled
    if lightning_module.use_reward_shaping:
        metrics.update(
            {
                f"{prefix}/mean_raw_reward": raw_rewards_tensor.mean(),
                f"{prefix}/std_raw_reward": raw_rewards_tensor.std(),
            }
        )

    # Log length reward statistics if target length is enabled
    if lightning_module.use_target_length:
        metrics.update(
            {
                f"{prefix}/mean_length_reward": length_rewards_tensor.mean(),
                f"{prefix}/std_length_reward": length_rewards_tensor.std(),
            }
        )

    # Log entropy reward statistics if target entropy is enabled
    if lightning_module.use_target_entropy:
        metrics.update(
            {
                f"{prefix}/mean_entropy_reward": entropy_rewards_tensor.mean(),
                f"{prefix}/std_entropy_reward": entropy_rewards_tensor.std(),
            }
        )

    lightning_module.log_dict(
        metrics,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
        prog_bar=True,
        logger=True,
    )

    # Print rewards and sequence lengths on this step
    print(
        f"Step {lightning_module.global_step}: Mean reward: {rewards_tensor.mean().item():.4f}, Std: {rewards_tensor.std().item():.4f}"
    )
    print(
        f"Step {lightning_module.global_step}: Mean raw reward: {raw_rewards_tensor.mean().item():.4f}, Std raw: {raw_rewards_tensor.std().item():.4f}"
    )
    print(
        f"Step {lightning_module.global_step}: Mean seq length: {seq_lengths_tensor.mean().item():.2f}, Std seq length: {seq_lengths_tensor.std().item():.2f}"
    )
    print(
        f"Step {lightning_module.global_step}: Mean entropy: {entropies_tensor.mean().item():.4f}, Std entropy: {entropies_tensor.std().item():.4f}"
    )
    if lightning_module.use_target_length:
        print(
            f"Step {lightning_module.global_step}: Mean length reward: {length_rewards_tensor.mean().item():.4f}, Std length reward: {length_rewards_tensor.std().item():.4f}"
        )
    if lightning_module.use_target_entropy:
        print(
            f"Step {lightning_module.global_step}: Mean entropy reward: {entropy_rewards_tensor.mean().item():.4f}, Std entropy reward: {entropy_rewards_tensor.std().item():.4f}"
        )

    # Print some randomly chosen generated sequences as diagnostic
    scores.print_example_sequences(
        generated_sequences,
        rewards_tensor,
        raw_rewards_tensor,
        lightning_module.token_info,
        lightning_module.global_step,
        num_examples=5,
    )
    return rewards_tensor


def _compute_and_log_advantages(lightning_module, tokens, rewards_tensor, prefix):
    # Calculate normalized relative advantage within groups
    batch_size = tokens.shape[0]
    group_size = lightning_module.group_size.item()

    # Reshape rewards to (batch_size, group_size) for group-wise statistics (that's along axis 1)
    rewards_grouped = rewards_tensor.view(batch_size, group_size)  # Shape (B, G)

    # Compute group-wise mean and standard deviation
    group_means = rewards_grouped.mean(dim=1, keepdim=True)  # (B, 1)
    group_stds = rewards_grouped.std(dim=1, keepdim=True, unbiased=False)  # (B, 1)

    # Add small epsilon to prevent div by zero and handle identical rewards
    epsilon = 1e-8
    group_stds = torch.clamp(group_stds, min=epsilon)

    # Calculate advantages based on normalize_advantage flag
    if lightning_module.normalize_advantage:
        # Calculate advantages = (reward - group_mean) / group_std (normalized)
        advantages = (
            rewards_grouped - group_means
        ) / group_stds  # (batch_size, group_size)
    else:
        # Calculate advantages as in Dr. GRPO, where no advantage normalization is done
        advantages = rewards_grouped - group_means  # (batch_size, group_size)

    # Flatten back to match original reward structure
    advantages = advantages.view(-1)  # (B * G, )

    # Log advantage stats
    advantage_metrics = {
        f"{prefix}/mean_advantage": advantages.mean(),
        f"{prefix}/std_advantage": advantages.std(),
    }
    lightning_module.log_dict(
        advantage_metrics,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
        prog_bar=True,
        logger=True,
    )

    # Print advantage on step
    print(
        f"Step {lightning_module.global_step}: Mean advantage: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}, Min: {advantages.min().item():.4f}, Max: {advantages.max().item():.4f}"
    )
    return advantages


def _compute_policy_logps_lists(lightning_module, generated_sequences):
    # Calculate per-token logps
    logp_start = time.time()
    per_token_logps_list = []
    ref_per_token_logps_list = []
    for seq_data in generated_sequences:  # This is sequential right now
        sequence = seq_data["sequence"].unsqueeze(
            dim=0
        )  # Tokens, contains bos, eos, and pad
        attn_mask = seq_data["seq_ids"].unsqueeze(
            dim=0
        )  # For logp calculation, use seq_ids attn mask
        structure = None

        # Compute current policy logp
        per_token_logps = compute_policy_logps(
            lightning_module.model, sequence, structure, attn_mask
        )
        per_token_logps_list.append(per_token_logps)

        # Compute reference policy logp for D_KL
        ref_per_token_logps = compute_policy_logps(
            lightning_module.reference_model, sequence, structure, attn_mask
        )
        ref_per_token_logps_list.append(ref_per_token_logps)

    logp_time = time.time() - logp_start
    print(
        f"Step {lightning_module.global_step}: Logp calculation took {logp_time:.3f}s"
    )
    return per_token_logps_list, ref_per_token_logps_list


def _compute_and_log_grpo_loss(
    lightning_module,
    generated_sequences,
    advantages,
    per_token_logps_list,
    ref_per_token_logps_list,
    prefix,
):
    # Calculate GRPO loss
    group_losses = {}
    total_kl_sum = 0.0  # Accumulate sum of KL values for batch-level logging
    total_response_tokens = 0  # Count total number of response tokens
    total_advantage_sum = 0.0
    total_kl_penalty_sum = 0.0
    total_tokens_for_components = 0

    # This loop runs calculations per-response
    for i, seq_data in enumerate(generated_sequences):
        # sequence = seq_data["sequence"]
        response_mask = seq_data["response_mask"]
        prompt_idx = seq_data["prompt_idx"]
        advantage = advantages[i]

        # The following logps are 1 shorter than len(sequence) and len(response_mask)
        per_token_logps = per_token_logps_list[i]
        ref_per_token_logps = ref_per_token_logps_list[i]

        # Compute per-token advantages with PPO-style clipping
        # Calculate policy ratio (the term multiplying advantage)
        ratio = torch.exp(per_token_logps - per_token_logps.detach())

        # Clip the ratio to [1 - epsilon_clip, 1 + epsilon_clip]
        ratio_clipped = torch.clamp(
            ratio,
            1.0 - lightning_module.epsilon_clip,
            1.0 + lightning_module.epsilon_clip,
        )

        # Compute unclipped and clipped objectives
        unclipped_advantages = ratio * advantage
        clipped_advantages = ratio_clipped * advantage

        # Take min
        per_token_advantages = torch.min(unclipped_advantages, clipped_advantages)

        # Calculate per-token D_KL (Schulman approximation)
        # http://joschu.net/blog/kl-approx.html
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        kl_penalty_term = lightning_module.beta_kl * per_token_kl

        # Accumulate KL and component magnitude values for batch-level logging
        masked_response = response_mask[1:]
        masked_per_token_kl = per_token_kl * masked_response
        total_kl_sum += masked_per_token_kl.sum().item()
        num_response_tokens = masked_response.sum().item()
        total_response_tokens += num_response_tokens

        total_advantage_sum += (
            (per_token_advantages * masked_response).abs().sum().item()
        )
        total_kl_penalty_sum += (kl_penalty_term * masked_response).abs().sum().item()
        total_tokens_for_components += num_response_tokens

        per_token_loss = -(per_token_advantages - kl_penalty_term)
        # Entropy now in reward, in advantages. Also, ratio is already folded into advantages here (see above)

        # Pass through response_mask
        per_token_loss = per_token_loss * masked_response

        # Sum loss over response
        # This is sum_{t=1}^{|o_i|} A - beta * D_KL (per token values)
        loss = per_token_loss.sum(dim=1)
        # loss = per_token_loss.sum(dim=1) / response_mask.sum() # Divided by response length

        # Group per-response loss by prompt_idx (G)
        if prompt_idx not in group_losses:
            group_losses[prompt_idx] = []
        group_losses[prompt_idx].append(loss)
        # Here, the structure of group_losses is:
        # {prompt_idx: [loss for response1, ... , loss for responseG]
        # e.g., if batch_size=2 and G=4, then group_losses has keys 0, 1
        # and for each of those keys, outputs lists of len(4)

    # Collect all per-token losses and completion masks
    all_per_token_losses = []
    all_response_masks = []

    for prompt_idx in sorted(group_losses.keys()):
        losses = group_losses[prompt_idx]
        for loss in losses:
            all_per_token_losses.append(loss)  # (B * G, )

    for i, seq_data in enumerate(generated_sequences):
        response_mask = seq_data["response_mask"]
        all_response_masks.append(response_mask)

    # Stack/pad all losses and masks
    all_per_token_losses = torch.stack(
        all_per_token_losses
    )  # (batch_size * group_size, max_seq_len)
    try:
        all_response_masks = torch.stack(
            all_response_masks
        )  # (batch_size * group_size, max_seq_len)
    except RuntimeError:
        all_response_masks = pad_sequence(
            all_response_masks, batch_first=True, padding_value=0
        )

    # DAPO loss: L = -1/N * sum(per_token_loss * completion_mask)
    # where N is total completion tokens across entire batch
    total_completion_tokens = all_response_masks.sum().clamp(min=1.0)
    grpo_batch_loss = (
        all_per_token_losses * all_response_masks
    ).sum() / total_completion_tokens

    # Calculate batch-level KL divergence for logging
    # Compute mean by dividing accumulated sum by total number of response tokens
    batch_kl_divergence = total_kl_sum / total_response_tokens
    # Convert to tensor for logging
    batch_kl_divergence = torch.tensor(
        batch_kl_divergence, device=lightning_module.device
    )

    # Convert component magnitude accumulators to tensors for logging
    mean_advantage_magnitude = torch.tensor(
        total_advantage_sum / total_tokens_for_components,
        device=lightning_module.device,
    )
    mean_kl_penalty_magnitude = torch.tensor(
        total_kl_penalty_sum / total_tokens_for_components,
        device=lightning_module.device,
    )

    # Log GRPO loss and KL divergence
    grpo_metrics = {
        f"{prefix}/loss_grpo": grpo_batch_loss,
        f"{prefix}/kl_divergence": batch_kl_divergence,
        f"{prefix}/loss_advantage_magnitude": mean_advantage_magnitude,
        f"{prefix}/loss_kl_penalty_magnitude": mean_kl_penalty_magnitude,
    }

    lightning_module.log_dict(
        grpo_metrics,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
        prog_bar=True,
        logger=True,
    )

    # Print loss and KL on step
    print(
        f"Step {lightning_module.global_step}: GRPO Loss: {grpo_batch_loss.item():.4f}, KL Div: {batch_kl_divergence.item():.4f}"
    )
    return grpo_batch_loss


def shared_eval_grpo(lightning_module, batch, batch_idx, prefix):
    """
    Implementation of GRPO with DAPO modification. See reference here https://huggingface.co/papers/2503.14476

    Args:
        lightning_module: LightningModule instance with model and hyperparameters
        batch: Provides tuples of prompt (tokens, masks) from TransformerOnlineDataset
        batch_idx: Batch index
        prefix: Logging prefix ("train", "validation", or "test")

    Returns:
        GRPO objective loss
    """

    start_time = time.time()

    # Unpack batch from Dataset
    tokens, masks = batch  # sequence_id attention masks

    _initialize_grpo_config(lightning_module)

    # Assign special tokens
    _pad_token = lightning_module.token_info["input"]["TOK"]["TOK_PAD"]
    _stop_token = lightning_module.token_info["input"]["TOK"]["TOK_STOP"]
    _start_token = lightning_module.token_info["input"]["TOK"]["TOK_START"]

    # Generate sequences (in total, batch_size * group_size). Sequences here are in tokens
    generation_start = time.time()
    generated_sequences, num_invalid_sequences = generate_sequences_online(
        lightning_module, tokens, masks, lightning_module.group_size
    )
    generation_time = time.time() - generation_start
    print(
        f"Step {lightning_module.global_step}: Sequence generation took {generation_time:.3f}s"
    )

    _compute_and_log_percent_identity(lightning_module, generated_sequences, prefix)
    rewards_tensor = _compute_and_log_rewards(
        lightning_module, generated_sequences, num_invalid_sequences, prefix
    )
    advantages = _compute_and_log_advantages(
        lightning_module, tokens, rewards_tensor, prefix
    )
    per_token_logps_list, ref_per_token_logps_list = _compute_policy_logps_lists(
        lightning_module, generated_sequences
    )
    grpo_batch_loss = _compute_and_log_grpo_loss(
        lightning_module,
        generated_sequences,
        advantages,
        per_token_logps_list,
        ref_per_token_logps_list,
        prefix,
    )

    total_time = time.time() - start_time
    print(
        f"Step {lightning_module.global_step}: Total GRPO step took {total_time:.3f}s"
    )

    return grpo_batch_loss
