import time
import lightning as L
import torch
import torch.nn as nn
import pl_bolts
from torch.nn.utils.rnn import pad_sequence
from lightning.pytorch.utilities import grad_norm

from idr_plm.nn.transformer.nn import GeometricMolTransformer
from idr_plm.utils.sampler import TokenSampler
from idr_plm.nn.transformer.scores import (
    compute_fraction_alanine,
    compute_charge_kappa,
    compute_protgps_score,
    apply_quadratic_reward_shaping,
    print_example_sequences,
    calculate_percent_identities,
    calculate_idr_length,
    compute_length_reward,
    compute_sequence_entropy,
    compute_entropy_reward,
    valid_sequence_characters,
)


class LightningModel(L.LightningModule):
    def __init__(self, model_args: dict, token_info: dict, training_args: dict):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        try:
            assert self.training_args["training_mode"] in [
                "autoregressive",
                "grpo",
            ]
        except AssertionError:
            raise ValueError("Invalid training mode specified")
        self.token_info = token_info
        models = self.build_model_base()
        if self.training_args["training_mode"] in ("grpo"):
            self.reference_model, self.model = models
        else:
            self.model = models
            self.reference_model = None

        if self.model_args["load_model"] is not None:
            self.load_model_from_checkpoint(self.model_args["load_model"])

        # Extract token_sampler args for online training modes that need sequence generation
        # NOTE: these lines are needed in the training config:
        # "++training.lightning_model_args.sampler_args.method=full"\
        # "++training.lightning_model_args.sampler_args.sample_val=1"\
        # "++training.lightning_model_args.sampler_args.temperature=1"\
        lightning_model_args = self.training_args.get("lightning_model_args", {})
        sampler_args = lightning_model_args.get("sampler_args", {})
        self.token_limit = sampler_args.get("token_limit", 1000)
        if sampler_args and self.training_args["training_mode"] in ("grpo",):
            # Use sampler_args configs to make TokenSampler
            self.sampling_method = sampler_args.get("method", "full")
            self.sample_val = sampler_args.get("sample_val", 1)
            self.sampling_temperature = sampler_args.get("temperature", 1.0)
            self.token_sampler = TokenSampler(
                method=self.sampling_method,
                sample_val=self.sample_val,
                temperature=self.sampling_temperature,
            )
        else:
            self.token_sampler = None

        self.loss_fn = self.build_loss_fn()

        # Initialize inference iteration counter for debugging
        self._inference_iteration = 0

        self.save_hyperparameters()

    def build_model_base(self):
        if self.training_args["training_mode"] in ("grpo"):
            # Create current model
            if self.model_args["model"] == "GeometricMolTransformer":
                current_model = GeometricMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                )
            else:
                raise ValueError("Invalid model type specified")

            # Create reference model
            if self.model_args["model"] == "GeometricMolTransformer":
                reference_model = GeometricMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                )

            else:
                raise ValueError("Invalid model type specified")

            # reference_model.load_state_dict(current_model.state_dict())
            # # Freeze reference model parameters
            # for param in reference_model.parameters():
            #     param.requires_grad = False
            # reference_model.eval()
            return [reference_model, current_model]

        else:  # Standard autoregressive training mode
            if self.model_args["model"] == "GeometricMolTransformer":
                return GeometricMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                )

            else:
                raise ValueError("Invalid model type specified")

    def build_loss_fn(self):
        if self.training_args["training_mode"] in ["autoregressive"]:
            loss_fn_base = nn.CrossEntropyLoss
            if self.training_args["loss_fn_args"] is not None:
                loss_fn = loss_fn_base(**self.training_args["loss_fn_args"])
            else:
                loss_fn = loss_fn_base(
                    reduction="none", ignore_index=self.smi_info["pad"]
                )
            return loss_fn

        elif self.training_args["training_mode"] in [
            "grpo",
        ]:
            return None

    def load_model_from_checkpoint(self, filename):
        print(f"Loading model from ckpt file {filename}")
        # Load checkpoint to CPU first
        # In GRPO this seems necessary to avoid GPU conflicts when training in DDP mode. This resolves the "gpu is busy" error when trying to load models from checkpoints
        ckpt = torch.load(filename, map_location="cpu")["state_dict"]
        # ckpt = torch.load(filename)["state_dict"]

        # Remove the model. prefix from all checkpoint keys
        ckpt = {".".join(k.split(".")[1:]): v for k, v in ckpt.items()}
        curr_state_dict = self.model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in ckpt.items()
            if (k in curr_state_dict) and curr_state_dict[k].shape == v.shape
        }

        if self.training_args["training_mode"] in ("grpo"):
            # Load pre-trained model for post-training
            self.model.load_state_dict(pretrained_dict, strict=False)
            # Load reference model
            self.reference_model.load_state_dict(pretrained_dict, strict=False)
            # Reference model: disable grad and set to eval mode
            for param in self.reference_model.parameters():
                param.requires_grad = False

            for p in self.reference_model.parameters():
                assert not p.requires_grad
            for p in self.model.parameters():
                assert p.requires_grad

            self.reference_model.eval()
            # NOTE: For now, this needs to be added to the training config to account for the frozen reference model during training:
            # "training.trainer_args.strategy=ddp_find_unused_parameters_true"\
            # https://github.com/Lightning-AI/pytorch-lightning/issues/17212

        else:
            # For all other training modes, just load model
            self.model.load_state_dict(pretrained_dict, strict=False)

    def configure_optimizers(self):
        u_optimizer = getattr(
            torch.optim, self.hparams.training_args.lightning_model_args.optimizer
        )
        u_optimizer = u_optimizer(
            self.model.parameters(),
            **self.hparams.training_args.lightning_model_args.optimizer_args,
        )
        if self.hparams.training_args.lightning_model_args.lr_scheduler is None:
            to_return = {"optimizer": u_optimizer}
        else:
            if (
                self.hparams.training_args.lightning_model_args.lr_scheduler
                == "LinearWarmupCosineAnnealingLR"
            ):
                u_scheduler = getattr(
                    pl_bolts.optimizers.lr_scheduler,
                    self.hparams.training_args.lightning_model_args.lr_scheduler,
                )
            else:
                u_scheduler = getattr(
                    torch.optim.lr_scheduler,
                    self.hparams.training_args.lightning_model_args.lr_scheduler,
                )
            u_scheduler = u_scheduler(
                u_optimizer,
                **self.hparams.training_args.lightning_model_args.lr_scheduler_args,
            )
            lr_scheduler_config = {
                "scheduler": u_scheduler,
                "interval": self.hparams.training_args.lightning_model_args.interval,
                "monitor": self.hparams.training_args.lightning_model_args.lr_scheduler_monitor,
            }
            to_return = {"optimizer": u_optimizer, "lr_scheduler": lr_scheduler_config}
        return to_return

    def _shared_eval_autoreg(self, batch, batch_idx, prefix):
        """Autoregressive shared_eval has no loss masking"""
        struct, src, src_key_pad_mask, tgt, tgt_key_pad_mask, seq_id = batch
        out = self.model(src, struct, seq_id)
        out = out.permute(0, 2, 1)
        loss = self.loss_fn(out, tgt)
        loss = loss.mean()
        metrics = {f"{prefix}/loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def _compute_policy_logps(self, model, tokens, structure, masks):
        """Calculate per-token logps under a provided model"""
        if isinstance(model, GeometricMolTransformer):
            policy_logits = model(tokens, structure, masks)

        policy_logps = policy_logits.log_softmax(dim=-1)
        policy_logps = torch.gather(
            policy_logps[:, :-1, :], dim=-1, index=tokens[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        # policy_logps is going to be 1 shorter than len(tokens) because start token doesn't have a logp
        return policy_logps

    def _shared_eval_grpo(self, batch, batch_idx, prefix):
        """
        Group relative policy optimization (GRPO) evaluation with DAPO token-level loss. See reference here https://huggingface.co/papers/2503.14476

        Args:
            batch: Provides tuples of prompt (tokens, masks) from TransformerOnlineDataset
            batch_idx: Batch index
            prefix: Logging prefix ("train", "validation", or "test")

        External args: (Passed in through hydra config)
            group_size: Number of sequences to generate per prompt
            epsilon_clip: PPO-style clipping ratio
            mu_grpo: Number of optimization steps to take for a batch of sampled prompts + completions
            beta_kl: Relative magnitude of D_KL in loss

        Returns:
            GRPO objective loss
        """

        start_time = time.time()

        # Unpack batch from Dataset
        tokens, masks = batch  # sequence_id attention masks

        # Assign GRPO variables from lightning_model_args (hydra ++ params)
        if not hasattr(self, "group_size"):
            self.group_size = torch.tensor(
                self.hparams.training_args.lightning_model_args.get("group_size", 4),
                device=self.device,
            )

        if not hasattr(self, "epsilon_clip"):
            self.epsilon_clip = torch.tensor(
                self.hparams.training_args.lightning_model_args.get(
                    "epsilon_clip", 0.2
                ),
                device=self.device,
            )

        if not hasattr(self, "mu_grpo"):
            self.mu_grpo = torch.tensor(
                self.hparams.training_args.lightning_model_args.get("mu_grpo", 1),
                device=self.device,
            )

        if not hasattr(self, "beta_kl"):
            self.beta_kl = torch.tensor(
                self.hparams.training_args.lightning_model_args.get("beta_kl", 0.1),
                device=self.device,
            )

        # Reward shaping hyperparameters
        if not hasattr(self, "use_reward_shaping"):
            self.use_reward_shaping = (
                self.hparams.training_args.lightning_model_args.get(
                    "use_reward_shaping", False
                )
            )

        if not hasattr(self, "reward_target_value"):
            self.reward_target_value = torch.tensor(
                self.hparams.training_args.lightning_model_args.get(
                    "reward_target_value", 0.5
                ),
                device=self.device,
            )

        if not hasattr(self, "reward_scale"):
            self.reward_scale = torch.tensor(
                self.hparams.training_args.lightning_model_args.get(
                    "reward_scale", 1.0
                ),
                device=self.device,
            )

        # Percent identity calculation sampling fraction
        if not hasattr(self, "pid_sample_fraction"):
            self.pid_sample_fraction = (
                self.hparams.training_args.lightning_model_args.get(
                    "pid_sample_fraction", 1.0
                )
            )

        # Percent identity penalty for diversity
        if not hasattr(self, "pid_penalty"):
            self.pid_penalty = torch.tensor(
                self.hparams.training_args.lightning_model_args.get("pid_penalty", 0.0),
                device=self.device,
            )

        # Advantage normalization flag
        if not hasattr(self, "normalize_advantage"):
            self.normalize_advantage = (
                self.hparams.training_args.lightning_model_args.get(
                    "normalize_advantage", True
                )
            )

        # Entropy reward parameters
        if not hasattr(self, "use_target_entropy"):
            self.use_target_entropy = (
                self.hparams.training_args.lightning_model_args.get(
                    "use_target_entropy", False
                )
            )

        if not hasattr(self, "target_entropy"):
            self.target_entropy = torch.tensor(
                self.hparams.training_args.lightning_model_args.get(
                    "target_entropy", 2.5
                ),
                device=self.device,
            )

        if not hasattr(self, "entropy_reward_weight"):
            self.entropy_reward_weight = torch.tensor(
                self.hparams.training_args.lightning_model_args.get(
                    "entropy_reward_weight", 0.1
                ),
                device=self.device,
            )

        if not hasattr(self, "entropy_reward_width"):
            self.entropy_reward_width = (
                self.hparams.training_args.lightning_model_args.get(
                    "entropy_reward_width", 1.0
                )
            )

        # Reward function selection
        if not hasattr(self, "reward_function_name"):
            self.reward_function_name = (
                self.hparams.training_args.lightning_model_args.get(
                    "reward_function_name", "compute_charge_kappa"
                )
            )

        # ProtGPS-specific parameters
        if not hasattr(self, "protgps_target_compartment"):
            self.protgps_target_compartment = (
                self.hparams.training_args.lightning_model_args.get(
                    "protgps_target_compartment", "p-body"
                )
            )
        if not hasattr(self, "protgps_parent_dir"):
            self.protgps_parent_dir = (
                self.hparams.training_args.lightning_model_args.get(
                    "protgps_parent_dir", "/home/protgps"
                )
            )
        if not hasattr(self, "protgps_aggregation"):
            self.protgps_aggregation = (
                self.hparams.training_args.lightning_model_args.get(
                    "protgps_aggregation", "max"
                )
            )

        # Target length reward parameters
        if not hasattr(self, "use_target_length"):
            self.use_target_length = (
                self.hparams.training_args.lightning_model_args.get(
                    "use_target_length", False
                )
            )

        if not hasattr(self, "target_length"):
            self.target_length = torch.tensor(
                self.hparams.training_args.lightning_model_args.get(
                    "target_length", 100
                ),
                device=self.device,
            )

        if not hasattr(self, "length_reward_weight"):
            self.length_reward_weight = torch.tensor(
                self.hparams.training_args.lightning_model_args.get(
                    "length_reward_weight", 0.1
                ),
                device=self.device,
            )

        if not hasattr(self, "length_reward_width"):
            # Controls the width of the length reward curve. Higher values allow more variability.
            # Default 1.0 is tight; increase to 2.0, 3.0 etc for wider tolerance around target length
            self.length_reward_width = (
                self.hparams.training_args.lightning_model_args.get(
                    "length_reward_width", 1.0
                )
            )

        # Map reward function name to actual function
        reward_function_map = {
            "compute_charge_kappa": compute_charge_kappa,
            "compute_fraction_alanine": compute_fraction_alanine,
            # "compute_qed_score": compute_qed_score,
            "compute_protgps_score": compute_protgps_score,
        }
        self.reward_function = reward_function_map[self.reward_function_name]

        # Assign special tokens
        _pad_token = self.token_info["input"]["TOK"]["TOK_PAD"]
        _stop_token = self.token_info["input"]["TOK"]["TOK_STOP"]
        _start_token = self.token_info["input"]["TOK"]["TOK_START"]

        # Generate sequences (in total, batch_size * group_size). Sequences here are in tokens
        generation_start = time.time()
        generated_sequences, num_invalid_sequences = self._generate_sequences_online(
            tokens, masks, self.group_size
        )
        generation_time = time.time() - generation_start
        print(
            f"Step {self.global_step}: Sequence generation took {generation_time:.3f}s"
        )

        # Calculate all-to-all percent identities using biopython
        percent_identity_start = time.time()

        percent_identities = calculate_percent_identities(
            generated_sequences,
            token_info=self.token_info,
            global_align=False,  # When False, normalize by min(len(seq1), len(seq2))
            sample_fraction=self.pid_sample_fraction,
        )

        percent_identity_time = time.time() - percent_identity_start
        print(
            f"Step {self.global_step}: Percent identity calculation took {percent_identity_time:.3f}s"
        )

        # Calculate and log percent IDs
        percent_identities_tensor = torch.tensor(percent_identities, device=self.device)
        mean_pid = percent_identities_tensor.mean().item()
        std_pid = percent_identities_tensor.std().item()

        # Convert mean_pid to tensor for use in loss calculation
        mean_pid_tensor = torch.tensor(mean_pid, device=self.device)

        print(
            f"Step {self.global_step}: Mean percent identity: {mean_pid:.2f}%, Std: {std_pid:.2f}%"
        )

        # Log to metrics
        pid_metrics = {
            f"{prefix}/mean_percent_identity": mean_pid,
            f"{prefix}/std_percent_identity": std_pid,
        }
        self.log_dict(
            pid_metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )

        # Compute rewards for generated sequences
        # As a placeholder reward, run compute_fraction_alanine() on sequences
        reward_start = time.time()
        rewards = []
        raw_rewards = []  # Store raw rewards before shaping
        length_rewards = []  # Store length rewards separately for logging
        entropy_rewards = []  # Store entropy rewards separately for logging
        entropies = []  # Store entropy values for logging
        for seq_data in generated_sequences:  # Reward calculation is serial here
            # Extract the sequence tensor
            sequence = seq_data["sequence"]  # tokens
            # Sequences are passed into score as tokens
            # Here, sequences will contain start_token, stop_token, and pad_tokens
            # Alphabet is contained in self.token_info

            # Call reward function with appropriate parameters
            if self.reward_function_name == "compute_protgps_score":
                raw_reward = self.reward_function(
                    sequence,
                    self.token_info,
                    self.device,
                    target_compartment=self.protgps_target_compartment,
                    protgps_parent_dir=self.protgps_parent_dir,
                    aggregation=self.protgps_aggregation,
                )
            else:
                raw_reward = self.reward_function(
                    sequence, self.token_info, self.device
                )

            raw_rewards.append(raw_reward)

            # Apply reward shaping if enabled
            if self.use_reward_shaping:
                shaped_reward = apply_quadratic_reward_shaping(
                    raw_reward, self.reward_target_value, self.reward_scale
                )
            else:
                shaped_reward = raw_reward

            # Compute length reward if enabled and add to total reward
            if self.use_target_length:
                length_reward = compute_length_reward(
                    sequence,
                    self.token_info,
                    self.device,
                    target_length=self.target_length.item(),
                    length_reward_width=self.length_reward_width,
                )
                length_rewards.append(length_reward)
                total_reward = shaped_reward + self.length_reward_weight * length_reward
            else:
                length_rewards.append(torch.tensor(0.0, device=self.device))
                total_reward = shaped_reward

            # Compute entropy reward if enabled and add to total reward
            # Also compute entropy for logging regardless
            entropy = compute_sequence_entropy(sequence, self.token_info, self.device)
            entropies.append(entropy)
            if self.use_target_entropy:
                entropy_reward = compute_entropy_reward(
                    sequence,
                    self.token_info,
                    self.device,
                    target_entropy=self.target_entropy.item(),
                    entropy_reward_width=self.entropy_reward_width,
                )
                entropy_rewards.append(entropy_reward)
                total_reward = (
                    total_reward + self.entropy_reward_weight * entropy_reward
                )
            else:
                entropy_rewards.append(torch.tensor(0.0, device=self.device))

            rewards.append(total_reward)

        # Convert rewards to tensor
        rewards_tensor = torch.stack(rewards)
        raw_rewards_tensor = torch.stack(raw_rewards)
        length_rewards_tensor = torch.stack(length_rewards)
        entropy_rewards_tensor = torch.stack(entropy_rewards)
        entropies_tensor = torch.stack(entropies)
        reward_time = time.time() - reward_start
        print(f"Step {self.global_step}: Reward computation took {reward_time:.3f}s")

        # Calculate IDR lengths (excluding pad tokens)
        seq_lengths_tensor = calculate_idr_length(
            generated_sequences, self.token_info, self.device
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
        if self.use_reward_shaping:
            metrics.update(
                {
                    f"{prefix}/mean_raw_reward": raw_rewards_tensor.mean(),
                    f"{prefix}/std_raw_reward": raw_rewards_tensor.std(),
                }
            )

        # Log length reward statistics if target length is enabled
        if self.use_target_length:
            metrics.update(
                {
                    f"{prefix}/mean_length_reward": length_rewards_tensor.mean(),
                    f"{prefix}/std_length_reward": length_rewards_tensor.std(),
                }
            )

        # Log entropy reward statistics if target entropy is enabled
        if self.use_target_entropy:
            metrics.update(
                {
                    f"{prefix}/mean_entropy_reward": entropy_rewards_tensor.mean(),
                    f"{prefix}/std_entropy_reward": entropy_rewards_tensor.std(),
                }
            )

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )

        # Print rewards and sequence lengths on this step
        print(
            f"Step {self.global_step}: Mean reward: {rewards_tensor.mean().item():.4f}, Std: {rewards_tensor.std().item():.4f}"
        )
        print(
            f"Step {self.global_step}: Mean raw reward: {raw_rewards_tensor.mean().item():.4f}, Std raw: {raw_rewards_tensor.std().item():.4f}"
        )
        print(
            f"Step {self.global_step}: Mean seq length: {seq_lengths_tensor.mean().item():.2f}, Std seq length: {seq_lengths_tensor.std().item():.2f}"
        )
        print(
            f"Step {self.global_step}: Mean entropy: {entropies_tensor.mean().item():.4f}, Std entropy: {entropies_tensor.std().item():.4f}"
        )
        if self.use_target_length:
            print(
                f"Step {self.global_step}: Mean length reward: {length_rewards_tensor.mean().item():.4f}, Std length reward: {length_rewards_tensor.std().item():.4f}"
            )
        if self.use_target_entropy:
            print(
                f"Step {self.global_step}: Mean entropy reward: {entropy_rewards_tensor.mean().item():.4f}, Std entropy reward: {entropy_rewards_tensor.std().item():.4f}"
            )

        # Print 3 randomly chosen generated sequences
        print_example_sequences(
            generated_sequences,
            rewards_tensor,
            raw_rewards_tensor,
            self.token_info,
            self.global_step,
            num_examples=3,
        )

        # Calculate normalized relative advantage within groups
        batch_size = tokens.shape[0]
        group_size = self.group_size.item()

        # Reshape rewards to (batch_size, group_size) for group-wise statistics (that's along axis 1)
        rewards_grouped = rewards_tensor.view(batch_size, group_size)  # Shape (B, G)

        # Compute group-wise mean and standard deviation
        group_means = rewards_grouped.mean(dim=1, keepdim=True)  # (B, 1)
        group_stds = rewards_grouped.std(dim=1, keepdim=True, unbiased=False)  # (B, 1)

        # Add small epsilon to prevent division by zero and handle identical rewards
        epsilon = 1e-8
        group_stds = torch.clamp(group_stds, min=epsilon)

        # Calculate advantages based on normalize_advantage flag
        if self.normalize_advantage:
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
        self.log_dict(
            advantage_metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )

        # Print advantage on step
        print(
            f"Step {self.global_step}: Mean advantage: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}, Min: {advantages.min().item():.4f}, Max: {advantages.max().item():.4f}"
        )

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
            per_token_logps = self._compute_policy_logps(
                self.model, sequence, structure, attn_mask
            )
            per_token_logps_list.append(per_token_logps)

            # Compute reference policy logp for D_KL
            ref_per_token_logps = self._compute_policy_logps(
                self.reference_model, sequence, structure, attn_mask
            )
            ref_per_token_logps_list.append(ref_per_token_logps)

        logp_time = time.time() - logp_start
        print(f"Step {self.global_step}: Logp calculation took {logp_time:.3f}s")

        # Calculate GRPO loss
        group_losses = {}
        total_kl_sum = 0.0  # Accumulate sum of KL values for batch-level logging
        total_response_tokens = 0  # Count total number of response tokens

        # This loop runs calculations per-response
        for i, seq_data in enumerate(generated_sequences):
            sequence = seq_data["sequence"]
            response_mask = seq_data["response_mask"]
            prompt_idx = seq_data["prompt_idx"]
            advantage = advantages[i]

            # The following logps are 1 shorter than len(sequence) and len(response_mask)
            per_token_logps = per_token_logps_list[i]
            ref_per_token_logps = ref_per_token_logps_list[i]

            # Compute per-token advantages with PPO-style clipping
            # Calculate policy ratio: π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(per_token_logps - per_token_logps.detach())

            # Clip the ratio to [1 - epsilon_clip, 1 + epsilon_clip]
            ratio_clipped = torch.clamp(
                ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip
            )

            # Compute unclipped and clipped objectives
            unclipped_advantages = ratio * advantage
            clipped_advantages = ratio_clipped * advantage

            # Take minimum (pessimistic bound) as in PPO
            per_token_advantages = torch.min(unclipped_advantages, clipped_advantages)

            # Calculate per-token D_KL (Schulman approximation)
            # http://joschu.net/blog/kl-approx.html
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

            # Accumulate KL values for batch-level logging later
            masked_per_token_kl = per_token_kl * response_mask[1:]
            total_kl_sum += masked_per_token_kl.sum().item()
            total_response_tokens += response_mask[1:].sum().item()

            # Calculate per-token losses
            # Optimization minimizes loss, so will maximize within parenthesis, so will minimize PID and KL penalty
            # per_token_loss = -(per_token_advantages - self.beta_kl * per_token_kl)

            # Calculate individual loss components
            kl_penalty_term = self.beta_kl * per_token_kl
            pid_penalty_term = self.pid_penalty * mean_pid_tensor / 100

            per_token_loss = -(
                per_token_advantages - kl_penalty_term - pid_penalty_term
            )  # Include PID penalty (entropy now in reward)

            # Pass through response_mask
            per_token_loss = per_token_loss * response_mask[1:]

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

        # DAPO implementation, reference: https://huggingface.co/papers/2503.14476
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

        ##########

        # Calculate batch-level KL divergence for logging
        # Compute mean by dividing accumulated sum by total number of response tokens
        batch_kl_divergence = total_kl_sum / total_response_tokens
        # Convert to tensor for logging
        batch_kl_divergence = torch.tensor(batch_kl_divergence, device=self.device)

        # Calculate batch-level component magnitudes for logging
        total_advantage_sum = 0.0
        total_kl_penalty_sum = 0.0
        total_pid_penalty_sum = 0.0
        total_tokens_for_components = 0

        for i, seq_data in enumerate(generated_sequences):
            response_mask = seq_data["response_mask"]
            per_token_logps = per_token_logps_list[i]
            ref_per_token_logps = ref_per_token_logps_list[i]
            advantage = advantages[i]

            # Recalculate components for logging (with PPO-style clipping)
            ratio = torch.exp(per_token_logps - per_token_logps.detach())
            ratio_clipped = torch.clamp(
                ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip
            )
            unclipped_advantages = ratio * advantage
            clipped_advantages = ratio_clipped * advantage
            per_token_advantages = torch.min(unclipped_advantages, clipped_advantages)

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

            kl_penalty_term = self.beta_kl * per_token_kl
            pid_penalty_term = self.pid_penalty * mean_pid_tensor / 100

            # Apply response mask and accumulate
            masked_advantages = (per_token_advantages * response_mask[1:]).abs()
            masked_kl_penalty = (kl_penalty_term * response_mask[1:]).abs()
            masked_pid_penalty = (pid_penalty_term * response_mask[1:]).abs()

            total_advantage_sum += masked_advantages.sum().item()
            total_kl_penalty_sum += masked_kl_penalty.sum().item()
            total_pid_penalty_sum += masked_pid_penalty.sum().item()
            total_tokens_for_components += response_mask[1:].sum().item()

        # Convert to tensors for logging
        mean_advantage_magnitude = torch.tensor(
            total_advantage_sum / total_tokens_for_components, device=self.device
        )
        mean_kl_penalty_magnitude = torch.tensor(
            total_kl_penalty_sum / total_tokens_for_components, device=self.device
        )
        mean_pid_penalty_magnitude = torch.tensor(
            total_pid_penalty_sum / total_tokens_for_components, device=self.device
        )

        # Log GRPO loss and KL divergence
        grpo_metrics = {
            f"{prefix}/loss_grpo": grpo_batch_loss,
            f"{prefix}/kl_divergence": batch_kl_divergence,
            f"{prefix}/loss_advantage_magnitude": mean_advantage_magnitude,
            f"{prefix}/loss_kl_penalty_magnitude": mean_kl_penalty_magnitude,
            f"{prefix}/loss_pid_penalty_magnitude": mean_pid_penalty_magnitude,
        }

        self.log_dict(
            grpo_metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )

        # Print loss and KL on step
        print(
            f"Step {self.global_step}: GRPO Loss: {grpo_batch_loss.item():.4f}, KL Div: {batch_kl_divergence.item():.4f}"
        )

        total_time = time.time() - start_time
        print(f"Step {self.global_step}: Total GRPO step took {total_time:.3f}s")

        return grpo_batch_loss

    def _generate_sequences_online(self, tokens, masks, group_size):
        """Generate group_size completions for each prompt using forward_autoregressive_prompted

        Args:
            tokens: (batch_size, seq_len) - prompt tokens
            masks: (batch_size, seq_len) - attention masks for prompts
            group_size: int - number of completions to generate per prompt

        Returns:
            list: Generated sequences with completions for GRPO training
        """

        _pad_token = self.token_info["input"]["TOK"]["TOK_PAD"]
        _stop_token = self.token_info["input"]["TOK"]["TOK_STOP"]
        _start_token = self.token_info["input"]["TOK"]["TOK_START"]

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
        if isinstance(self.model, GeometricMolTransformer):
            structural_tokens = torch.zeros(
                (total_sequences, 1), dtype=torch.long, device=self.device
            )
        else:
            structural_tokens = torch.zeros(
                (total_sequences, 1), dtype=torch.long, device=self.device
            )

        # Convert tokens to list format expected by forward_autoregressive_prompted
        smi_batch_list = [repeated_tokens[i] for i in range(total_sequences)]

        # Generate completions for all prompts at once
        with torch.no_grad():
            # Create a TokenSampler for this generation if not available
            if hasattr(self, "token_sampler") and self.token_sampler is not None:
                sampler = self.token_sampler
            else:
                sampler = TokenSampler(
                    method=self.sampling_method,
                    sample_val=self.sample_val,
                    temperature=self.sampling_temperature,
                )

            # Generate completions using forward_autoregressive_prompted
            completed_sequences, completed_probs = self.forward_autoregressive_prompted(
                (structural_tokens, smi_batch_list, repeated_seq_id),
                token_sampler=sampler,
            )

        # Process all completed sequences and organize by prompt
        for i in range(total_sequences):
            if completed_sequences[i] is not None:
                # Calculate which prompt this sequence belongs to
                prompt_idx = i // group_size

                seq = torch.tensor(
                    completed_sequences[i], device=self.device
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

        # Initialize validity checking and recursion depth tracking
        if not hasattr(self, "check_sequence_validity"):
            self.check_sequence_validity = valid_sequence_characters

        if not hasattr(self, "log_invalid_sequences"):
            self.log_invalid_sequences = True

        if not hasattr(self, "_regeneration_depth"):
            self._regeneration_depth = 0
            self._max_regeneration_depth = 5

        # Check for invalid sequences
        invalid_indices = []
        for i, seq_data in enumerate(all_generated_sequences):
            if not self.check_sequence_validity(
                seq_data["sequence"], self.token_info, self.device
            ):
                invalid_indices.append(i)

        # Track the final number of invalid sequences
        final_num_invalid = len(invalid_indices)

        # Recursive regeneration if invalid sequences found
        if len(invalid_indices) > 0:
            if self._regeneration_depth < self._max_regeneration_depth:
                if self.log_invalid_sequences:
                    print(
                        f"Step {self.global_step}: Found {len(invalid_indices)} invalid sequences, recursing (depth {self._regeneration_depth + 1}/{self._max_regeneration_depth})"
                    )

                self._regeneration_depth += 1
                # Recursively call to regenerate all sequences
                sequences, num_invalid = self._generate_sequences_online(
                    tokens, masks, group_size
                )
                return sequences, num_invalid
            else:
                if self.log_invalid_sequences:
                    print(
                        f"Step {self.global_step}: {len(invalid_indices)} sequences remain invalid after max depth"
                    )
                self._regeneration_depth = 0
                raise RuntimeError(
                    f"Step {self.global_step}: Max regeneration depth ({self._max_regeneration_depth}) exceeded. "
                    f"{len(invalid_indices)} sequences remain invalid. Training halted to prevent propagation of invalid sequences."
                )
        else:
            # All valid, reset depth counter
            self._regeneration_depth = 0

        return all_generated_sequences, final_num_invalid

    def on_before_optimizer_step(self, optimizer):
        # Calculate the L2 norm of the gradients
        norms = grad_norm(self, norm_type=2)
        # Log the norms, for example, to TensorBoard
        self.log_dict(norms, prog_bar=False)

    def training_step(self, batch, batch_idx):
        if self.training_args["training_mode"] == "autoregressive":
            return self._shared_eval_autoreg(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "grpo":
            return self._shared_eval_grpo(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            if self.training_args["training_mode"] == "autoregressive":
                return self._shared_eval_autoreg(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "grpo":
                return self._shared_eval_grpo(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            if self.training_args["training_mode"] == "autoregressive":
                return self._shared_eval_autoreg(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "grpo":
                return self._shared_eval_grpo(batch, batch_idx, "test")

    def forward_autoregressive_prompted(self, batch, token_sampler):
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

        _pad_token = self.token_info["input"]["TOK"]["TOK_PAD"]
        _stop_token = self.token_info["input"]["TOK"]["TOK_STOP"]
        _start_token = self.token_info["input"]["TOK"]["TOK_START"]

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
        max_iterations = 1024  # See self.token_limit below too
        print()
        while len(completed_smiles_structures) < target_batch_size:
            iteration += 1
            self._inference_iteration = iteration

            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: {len(completed_smiles_structures)}/{target_batch_size} completed, batch_size={working_smi_batch.shape[0]}"
                )
            if iteration > max_iterations:
                print(f"WARNING: Reached max iterations ({max_iterations})")
                break
            # print(working_smi_batch)
            if self.training_args["training_mode"] == "era_online":
                logits = self.reference_model(
                    working_smi_batch,
                    working_structural_batch,
                    working_seq_id_batch,
                    batch_access_indices=all_indices,
                )  # (N, T, E)
            else:
                logits = self.model(
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

            # Check if next_pos is outside of self.token_limit
            # If it is, then append a stop token
            # NOTE: self.token_limit is now set in training config with:
            # "++training.lightning_model_args.sampler_args.token_limit=1000"\
            position_exceeds_limit = indexed_positions > self.token_limit
            tokens_to_insert = tokens.squeeze(-1).clone()
            tokens_to_insert[position_exceeds_limit] = _stop_token

            # FH: Here, indexed_positions is used as is because we want to update the first padding token
            # as the sampled non-padding token
            working_smi_batch[torch.arange(logits.shape[0]), indexed_positions] = (
                tokens_to_insert
            )
            working_token_probs = torch.cat(
                (working_token_probs, selected_probs), dim=1
            )

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

    def forward_autoregressive(self, batch, token_sampler):
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

        # mask_token = self.token_info['input']['TOK']['TOK_MASK']
        pad_token = self.token_info["input"]["TOK"]["TOK_PAD"]
        stop_token = self.token_info["input"]["TOK"]["TOK_STOP"]
        # start_token = self.token_info['input']['TOK']['TOK_START']

        while not all_structures_completed:
            print(
                "beginning of loop",
                working_smi_batch.shape,
                working_structural_batch.shape,
                working_seq_id_batch.shape,
            )
            logits = self.model(
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
                    torch.ones(
                        (working_seq_id_batch.shape[0], 1), device=device
                    ).long(),
                ),
                dim=1,
            )
            concatenated_structural = torch.cat(
                (
                    working_structural_batch,
                    torch.ones(
                        (working_structural_batch.shape[0], 1), device=device
                    ).long()
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
                working_smi_batch.shape[-1] > self.token_limit
            ):  # Fix this hardcoding for the token limit
                # NOTE: self.token_limit is now set in training config with:
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
                    completed_structures[idx] = (
                        working_smi_batch[j].detach().cpu().numpy()
                    )
                    completed_token_probs[idx] = (
                        working_token_probs[j].detach().cpu().numpy()
                    )

                all_structures_completed = True

            if len(working_smi_batch) == 0:
                all_structures_completed = True
            # import pdb; pdb.set_trace()

        return completed_structures, completed_token_probs


#####
# EXPOSED SAMPLING METHOD
#####


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
            tokens, probs = transformer_model.forward_autoregressive(
                (structural_tokens_batch, smiles_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        else:
            print("Generating sequences prompted")
            tokens, probs = transformer_model.forward_autoregressive_prompted(
                (structural_tokens_batch, smiles_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        all_sampled_tokens.append(tokens)
        all_token_probs.append(probs)
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch}/{num_batches} completed in {batch_time:.2f} seconds")

    return all_sampled_tokens, all_token_probs
