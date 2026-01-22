import time
import lightning as L
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pl_bolts
from torch.nn.utils.rnn import pad_sequence
import math
from lightning.pytorch.utilities import grad_norm

from idr_plm.nn.transformer.nn import (
    GeometricMolTransformer,
    # TransfusionMolTransformer,
    # SeqStructMixedTransformer,
)

# from idr_plm.nn.transformer.utils import get_node_features
# from idr_plm.nn.layers import diffusion
# from idr_plm.nn.layers.diffusion import sample_torsions
# from rdkit import Chem
# from rdkit import RDLogger
# from idr_plm.utils.structure import compute_dihedrals
from idr_plm.utils.sampler import TokenSampler
from idr_plm.nn.transformer.scores import (
    apply_gaussian_reward_shaping,
    # compute_qed_score,
    compute_fraction_alanine,
    compute_charge_kappa,
    compute_protgps_score,
    apply_quadratic_reward_shaping,
    print_example_sequences,
    apply_absolute_difference_reward_shaping,
    calculate_percent_identities,
    calculate_idr_length,
    compute_length_reward,
    compute_sequence_entropy,
    compute_entropy_reward,
    valid_sequence_characters,
)

# RDLogger.DisableLog("rdApp.*")


class LightningModel(L.LightningModule):
    def __init__(self, model_args: dict, token_info: dict, training_args: dict):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        try:
            assert self.training_args["training_mode"] in [
                "autoregressive",
                "bidirectional",
                "transfusion",
                "transfusion_offset",
                "era",
                "era_online",
                "grpo",
                "mixed_sequence",
                "dpo",
            ]
        except AssertionError:
            raise ValueError("Invalid training mode specified")
        self.token_info = token_info
        models = self.build_model_base()
        if self.training_args["training_mode"] in ("era_online", "grpo"):
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
        if sampler_args and self.training_args["training_mode"] in (
            "era_online",
            "grpo",
        ):
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
        if self.training_args["training_mode"] in ["transfusion_offset"]:
            assert self.training_args["manual_opt_args"] is not None
            # Number of steps that should happen before a ce_step happens
            assert "ce_n_steps" in self.training_args["manual_opt_args"]

        # Initialize inference iteration counter for debugging
        self._inference_iteration = 0

        self.save_hyperparameters()

    def build_model_base(self):
        if self.training_args["training_mode"] in ("era_online", "grpo"):
            # Create current model
            if self.model_args["model"] == "GeometricMolTransformer":
                current_model = GeometricMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                )
            elif self.model_args["model"] == "TransfusionMolTransformer":
                current_model = TransfusionMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                    structure_embedding_args=self.model_args["model_args"][
                        "structure_embedding_args"
                    ],
                    structure_out_dim=self.model_args["model_args"][
                        "structure_out_dim"
                    ],
                )
            elif self.model_args["model"] == "SeqStructMixedTransformer":
                current_model = SeqStructMixedTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                    embedding_args=self.model_args["model_args"]["embedding_args"],
                    forward_mode=self.model_args["model_args"]["forward_mode"],
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
            elif self.model_args["model"] == "TransfusionMolTransformer":
                reference_model = TransfusionMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                    structure_embedding_args=self.model_args["model_args"][
                        "structure_embedding_args"
                    ],
                    structure_out_dim=self.model_args["model_args"][
                        "structure_out_dim"
                    ],
                )
            elif self.model_args["model"] == "SeqStructMixedTransformer":
                reference_model = SeqStructMixedTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                    embedding_args=self.model_args["model_args"]["embedding_args"],
                    forward_mode=self.model_args["model_args"]["forward_mode"],
                )
            else:
                raise ValueError("Invalid model type specified")

            # reference_model.load_state_dict(current_model.state_dict())
            # # Freeze reference model parameters
            # for param in reference_model.parameters():
            #     param.requires_grad = False
            # reference_model.eval()
            return [reference_model, current_model]
        else:
            if self.model_args["model"] == "GeometricMolTransformer":
                return GeometricMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                )
            elif self.model_args["model"] == "TransfusionMolTransformer":
                return TransfusionMolTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                    structure_embedding_args=self.model_args["model_args"][
                        "structure_embedding_args"
                    ],
                    structure_out_dim=self.model_args["model_args"][
                        "structure_out_dim"
                    ],
                )
            elif self.model_args["model"] == "SeqStructMixedTransformer":
                return SeqStructMixedTransformer(
                    dim_model=self.model_args["model_args"]["d_model"],
                    token_info=self.token_info,
                    unified_transformer_args=self.model_args["model_args"][
                        "unified_transformer_args"
                    ],
                    embedding_args=self.model_args["model_args"]["embedding_args"],
                    forward_mode=self.model_args["model_args"]["forward_mode"],
                )
            else:
                raise ValueError("Invalid model type specified")

    def build_loss_fn(self):
        if self.training_args["training_mode"] in ["autoregressive", "bidirectional"]:
            loss_fn_base = nn.CrossEntropyLoss
            if self.training_args["loss_fn_args"] is not None:
                loss_fn = loss_fn_base(**self.training_args["loss_fn_args"])
            else:
                loss_fn = loss_fn_base(
                    reduction="none", ignore_index=self.smi_info["pad"]
                )
            return loss_fn
        elif self.training_args["training_mode"] in [
            "transfusion",
            "transfusion_offset",
        ]:
            assert self.training_args["loss_fn_args"] is not None
            loss_type = self.training_args["loss_fn"]
            assert loss_type in ["DDPMLoss", "TorDiffLoss"]
            # Get the loss function from the diffusion module in clm
            return getattr(diffusion, loss_type)(**self.training_args["loss_fn_args"])
        elif self.training_args["training_mode"] in [
            "era",
            "era_online",
            "mixed_sequence",
            "dpo",
            "grpo",
        ]:
            return None

    def load_model_from_checkpoint(self, filename):
        print(f"Loading model from ckpt file {filename}")
        # Load checkpoint to CPU first
        # In GRPO (and probably in era_online) this seems necessary to avoid GPU conflicts when training in DDP mode. This resolves the "gpu is busy" error when trying to load models from checkpoints
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

        if self.training_args["training_mode"] in ("era_online", "grpo"):
            # Load pre-trained model for post-training
            self.model.load_state_dict(pretrained_dict, strict=False)
            # Load reference model
            self.reference_model.load_state_dict(pretrained_dict, strict=False)
            # Reference model: disable grad and set to eval mode
            for param in self.reference_model.parameters():
                param.requires_grad = False

            for p in self.reference_model.parameters():
                assert p.requires_grad == False
            for p in self.model.parameters():
                assert p.requires_grad == True

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

    def clear_kv_cache(self):
        """Clear KV cache in the underlying model"""
        print("Cache cleared in module")
        if hasattr(self.model, "clear_kv_cache"):
            self.model.clear_kv_cache()
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "clear_kv_cache"
        ):
            self.model.transformer.clear_kv_cache()

    def _shared_eval_transfusion(self, batch, batch_idx, prefix):
        token_input, token_target, structure_segment, struct_mask, sequence_id = batch
        # Sample time steps for the noising
        losses = self.loss_fn(
            self.model,
            token_input,
            token_target,
            structure_segment,
            struct_mask,
            sequence_id,
        )
        # Returns each tensor separately to allow for separate optimization in a different
        #   shared eval function
        ce_loss, diff_loss, lambda_diff = losses
        loss_tensor = ce_loss + (lambda_diff * diff_loss)
        metrics = {
            f"{prefix}/loss": loss_tensor,
            f"{prefix}/ce": ce_loss,
            f"{prefix}/diff": diff_loss,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return loss_tensor

    def _shared_eval_transfusion_offset(self, batch, batch_idx, prefix):
        # TODO: How to handle learning rate scheduling in this case?
        token_input, token_target, structure_segment, struct_mask, sequence_id = batch
        # Sample time steps for the noising
        losses = self.loss_fn(
            self.model,
            token_input,
            token_target,
            structure_segment,
            struct_mask,
            sequence_id,
        )
        ce_loss, diff_loss, lambda_diff = losses
        if (batch_idx + 1) % self.hparams.training_args.manual_opt_args.ce_n_steps == 0:
            # Add in CE loss at specified intervals
            loss_to_opt = ce_loss + lambda_diff * diff_loss
        else:
            loss_to_opt = lambda_diff * diff_loss

        metrics = {
            f"{prefix}/loss": ce_loss + lambda_diff * diff_loss,
            f"{prefix}/ce": ce_loss,
            f"{prefix}/diff": diff_loss,
        }

        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return loss_to_opt

    def _shared_eval_bidirec(self, batch, batch_idx, prefix):
        """Bidirectional shared_eval has loss masking"""
        struct, smi, masked_smi, seq_id, loss_mask = batch
        out = self.model(masked_smi, struct, seq_id)
        out = out.permute(0, 2, 1)
        loss = self.loss_fn(out, smi)
        loss = (loss * loss_mask).sum(-1) / (loss_mask.sum(-1))
        loss = loss.mean()
        metrics = {f"{prefix}/loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
        return loss

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

    def _shared_eval_mixed_seq(self, batch, batch_idx, prefix):
        """Mixed sequence shared_eval with loss masking and the ability to support both
        structure and smiles tokens"""
        mode = batch[0]
        if mode == "smiles_only":
            _, smi_tokens, masked_smi_tokens, sequence_id, loss_mask = batch
            mod_out = self.model((mode, masked_smi_tokens, sequence_id))
            out, pad_token = mod_out
            out = out.permute(0, 2, 1)
            loss = nn.functional.cross_entropy(
                out, smi_tokens, reduction="none", ignore_index=pad_token
            )
            loss = (loss * loss_mask).sum(-1) / (loss_mask.sum(-1))
            loss = loss.mean()
            metrics = {f"{prefix}/loss": loss}
            self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
            return loss
        elif mode in ["smiles_and_struct", "smiles_struct_avh"]:
            (
                _,
                smi_tokens,
                masked_smi_tokens,
                struct_tokens,
                masked_struct_tokens,
                sequence_id,
                loss_mask_smi,
                loss_mask_struct,
            ) = batch
            out = self.model(
                (mode, masked_smi_tokens, masked_struct_tokens, sequence_id)
            )
            # smiles out and struct out both 2D here
            # smiles_out: (N, T, E)
            # struct_out: (?, E)
            smiles_out, struct_out, smiles_pad_token, struct_pad_token = out

            # SMILES loss calc
            smiles_loss = F.cross_entropy(
                smiles_out.reshape(-1, smiles_out.shape[-1]),
                smi_tokens.reshape(-1),
                ignore_index=smiles_pad_token,
                reduction="none",
            )
            loss_mask_smi_flat = loss_mask_smi.reshape(-1)
            smiles_loss = smiles_loss[loss_mask_smi_flat]
            smiles_loss = smiles_loss.mean()

            if struct_tokens.ndim == 3:
                # FH: Additional dimensions here if atom, valency, hybridization information is included, but only go off
                #   the first row in this scenario (which contains the true structure tokens)
                assert struct_tokens.shape[1] == 4
                struct_tokens = struct_tokens[:, 0, :]

            # Structure loss calc
            # GR: The structure token cross entropy loss is
            # computed over the non-padding tokens of the
            # structure track
            # FH: Have to be more nuanced here because the struct_out from the transformer
            # is only over non-padding structure tokens whereas the target struct_tokens and
            # loss_mask_struct take into account padding, so need to prune that out from both
            # the struct_tokns and loss_mask_struct to ensure the masks and target tokens are
            # applied properly

            struct_tokens_flat = struct_tokens.reshape(-1)
            loss_mask_struct_flat = loss_mask_struct.reshape(-1)
            non_padding_selection = struct_tokens_flat != struct_pad_token
            struct_tokens_flat = struct_tokens_flat[non_padding_selection]
            loss_mask_struct_flat = loss_mask_struct_flat[non_padding_selection]

            struct_loss = F.cross_entropy(
                struct_out,
                struct_tokens_flat,
                ignore_index=struct_pad_token,
                reduction="none",
            )
            struct_loss = struct_loss[loss_mask_struct_flat]
            struct_loss = struct_loss.mean()

            # Combine losses
            if torch.isnan(smiles_loss):
                smiles_loss = torch.tensor(0.0, device=smiles_loss.device)
            if torch.isnan(struct_loss):
                struct_loss = torch.tensor(0.0, device=struct_loss.device)

            loss = smiles_loss + struct_loss
            metrics = {
                f"{prefix}/loss": loss,
                f"{prefix}/smiles_loss": smiles_loss,
                f"{prefix}/struct_loss": struct_loss,
            }
            self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
            return loss
        elif mode == "ida":
            (
                _,
                smi_tokens,
                masked_smi_tokens,
                struct_tokens,
                masked_struct_tokens,
                coords,
                sequence_id,
                loss_mask_smi,
                loss_mask_struct,
            ) = batch
            out = self.model(
                (mode, masked_smi_tokens, masked_struct_tokens, coords, sequence_id)
            )
            # smiles out and struct out both 2D here
            # smiles_out: (N, T, E)
            # struct_out: (?, E)
            smiles_out, struct_out, smiles_pad_token, struct_pad_token = out

            # SMILES loss calc
            smiles_loss = F.cross_entropy(
                smiles_out.reshape(-1, smiles_out.shape[-1]),
                smi_tokens.reshape(-1),
                ignore_index=smiles_pad_token,
                reduction="none",
            )
            loss_mask_smi_flat = loss_mask_smi.reshape(-1)
            smiles_loss = smiles_loss[loss_mask_smi_flat]
            smiles_loss = smiles_loss.mean()

            # Structure loss calc
            if struct_tokens.ndim == 3:
                # FH: Additional dimensions here if atom, valency, hybridization information is included, but only go off
                #   the first row in this scenario (which contains the true structure tokens)
                assert struct_tokens.shape[1] == 4
                struct_tokens = struct_tokens[:, 0, :]

            # GR: The structure token cross entropy loss is
            # computed over the non-padding tokens of the
            # structure track
            # FH: Have to be more nuanced here because the struct_out from the transformer
            # is only over non-padding structure tokens whereas the target struct_tokens and
            # loss_mask_struct take into account padding, so need to prune that out from both
            # the struct_tokns and loss_mask_struct to ensure the masks and target tokens are
            # applied properly

            struct_tokens_flat = struct_tokens.reshape(-1)
            loss_mask_struct_flat = loss_mask_struct.reshape(-1)
            non_padding_selection = struct_tokens_flat != struct_pad_token
            struct_tokens_flat = struct_tokens_flat[non_padding_selection]
            loss_mask_struct_flat = loss_mask_struct_flat[non_padding_selection]

            struct_loss = F.cross_entropy(
                struct_out,
                struct_tokens_flat,
                ignore_index=struct_pad_token,
                reduction="none",
            )
            struct_loss = struct_loss[loss_mask_struct_flat]
            struct_loss = struct_loss.mean()

            # Combine losses
            if torch.isnan(smiles_loss):
                smiles_loss = torch.tensor(0.0, device=smiles_loss.device)
            if torch.isnan(struct_loss):
                struct_loss = torch.tensor(0.0, device=struct_loss.device)

            loss = smiles_loss + struct_loss
            metrics = {
                f"{prefix}/loss": loss,
                f"{prefix}/smiles_loss": smiles_loss,
                f"{prefix}/struct_loss": struct_loss,
            }
            self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True)
            return loss

    def _compute_policy_logps(self, model, tokens, structure, masks):
        """Calculate per-token logps under a provided model"""
        if isinstance(model, GeometricMolTransformer):
            policy_logits = model(tokens, structure, masks)
        elif isinstance(model, SeqStructMixedTransformer):
            policy_logits, _ = model(("smiles_only", tokens, masks))

        policy_logps = policy_logits.log_softmax(dim=-1)
        policy_logps = torch.gather(
            policy_logps[:, :-1, :], dim=-1, index=tokens[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        # policy_logps is going to be 1 shorter than len(tokens) because start token doesn't have a logp
        return policy_logps

    def _shared_eval_era(self, batch, batch_idx, prefix):
        if not hasattr(self, "alignment_betas"):
            self.alignment_betas = torch.tensor(
                self.hparams.training_args.lightning_model_args.beta
            ).to(self.device)

        tokens, masks, energies, ref_logps = batch

        policy_logps = self._compute_policy_logps(self.model, tokens, None, masks)
        # Ignore the BOS when computing logps
        policy_logps = policy_logps * masks[:, 1:]
        policy_logps = policy_logps.sum(-1)

        beta_prime = self.alignment_betas / (
            1 + self.hparams.training_args.lightning_model_args.gamma
        )
        gamma_prime = self.hparams.training_args.lightning_model_args.gamma / (
            1 + self.hparams.training_args.lightning_model_args.gamma
        )

        # FH: Here, energies can be (n_samples, n_energies) with associated beta_prime of shape (n_energies)
        # or energies can be (n_samples) with a scalar beta_prime. After the multiplication and summation
        # guarded behind the dimension check, energies should just be (n_samples).
        energies = energies * beta_prime
        if energies.dim() == 2:
            energies = energies.sum(-1)

        policy_logps_y1 = policy_logps.reshape(-1, 2)[:, 0]
        policy_logps_y2 = policy_logps.reshape(-1, 2)[:, 1]

        ref_logps_y1 = ref_logps.reshape(-1, 2)[:, 0]
        ref_logps_y2 = ref_logps.reshape(-1, 2)[:, 1]

        energies_y1 = energies.reshape(-1, 2)[:, 0]
        energies_y2 = energies.reshape(-1, 2)[:, 1]

        logp = nn.functional.logsigmoid(policy_logps_y2 - policy_logps_y1)
        logp_prime = nn.functional.logsigmoid(policy_logps_y1 - policy_logps_y2)

        logp_star = nn.functional.logsigmoid(
            -(energies_y2 - energies_y1) + (gamma_prime * (ref_logps_y2 - ref_logps_y1))
        )
        logp_star_prime = nn.functional.logsigmoid(
            -(energies_y1 - energies_y2) + (gamma_prime * (ref_logps_y1 - ref_logps_y2))
        )
        kl_loss = torch.exp(logp_star) * (logp_star - logp) + torch.exp(
            logp_star_prime
        ) * (logp_star_prime - logp_prime)

        kl_loss = kl_loss.mean()

        metrics = {f"{prefix}/ERALoss": kl_loss}

        self.log_dict(
            metrics,
            on_epoch=True,
            on_step=self.hparams.training_args.lightning_model_args.on_step,
            sync_dist=self.hparams.training_args.lightning_model_args.sync_dist,
            batch_size=tokens.shape[0],
        )

        return kl_loss

    def _shared_eval_era_online(self, batch, batch_idx, prefix):
        """
        Online ERA evaluation that samples from the current model instead of pre-computed pairs.

        Key differences from standard ERA:
        - Generates molecule pairs on-the-fly from the current model at each training step
        - Uses alanine fraction as a dummy reward function for demonstration purposes
        - Computes reference log probabilities using a reference model from previous training step
        - Reference model is automatically updated every N steps (default: 100, configurable via 'reference_update_frequency')
        - Multiplies KL loss by probability ratio: (π_θ(y)·π_θ(y')) / (π_ref(y)·π_ref(y'))

        Args:
            batch: Dummy batch data (actual generation happens in this method)
            batch_idx: Batch index
            prefix: Logging prefix ("train", "validation", or "test")

        Returns:
            KL divergence loss computed using same formula as standard ERA
        """
        tokens, masks = batch

        _pad_token = self.token_info["input"]["TOK"]["TOK_PAD"]

        if not hasattr(self, "alignment_betas"):
            self.alignment_betas = torch.tensor(
                self.hparams.training_args.lightning_model_args.beta
            ).to(self.device)

        # Arguments for what energy function to use
        if not hasattr(self, "energy_function"):
            self.energy_function = self.hparams.training_args.lightning_model_args.get(
                "energy_function", "fraction_alanine"
            )

        # Map energy function names to actual functions
        if not hasattr(self, "energy_function_map"):
            self.energy_function_map = {
                "fraction_alanine": compute_fraction_alanine,
                "charge_kappa": compute_charge_kappa,
            }

        # Reward shaping hyperparameters
        if not hasattr(self, "use_reward_shaping"):
            self.use_reward_shaping = (
                self.hparams.training_args.lightning_model_args.get(
                    "use_reward_shaping", False
                )
            )

        # Choose reward shaping function to use
        if self.use_reward_shaping:
            self.reward_shaping_function = (
                self.hparams.training_args.lightning_model_args.get(
                    "reward_shaping_function", "gaussian"
                )
            )

        # Map reward shaping function names to actual functions
        if not hasattr(self, "reward_shaping_function_map"):
            self.reward_shaping_function_map = {
                "gaussian": apply_gaussian_reward_shaping,
                "quadratic": apply_quadratic_reward_shaping,
                "absolute_difference": apply_absolute_difference_reward_shaping,
            }

        # Generate sample pairs from the current model
        generated_sequences = self._generate_sequences_online(
            tokens, masks, group_size=2
        )

        energies = []
        generated_tokens = []
        generated_masks = []
        raw_rewards = []
        for seq_data in generated_sequences:
            # Extract the sequence tensor
            sequence = seq_data["sequence"]
            reward = self.energy_function_map[self.energy_function](
                sequence, self.token_info, self.device
            )
            raw_rewards.append(reward)
            if self.use_reward_shaping:
                reward = self.reward_shaping_function_map[self.reward_shaping_function](
                    reward,
                    **self.hparams.training_args.lightning_model_args.get(
                        "reward_shaping_args", {}
                    ),
                )
            energies.append(reward)
            generated_tokens.append(sequence)
            generated_masks.append(seq_data["response_mask"])

        energies = torch.stack(energies).view(-1, 2)  # pairs: [batch_size, 2]
        raw_rewards_tensor = torch.stack(raw_rewards).view(
            -1, 2
        )  # pairs: [batch_size, 2]
        try:
            generated_tokens = torch.stack(
                generated_tokens
            )  # pairs: [batch_size * 2, seq_len]
            generated_masks = torch.stack(
                generated_masks
            )  # pairs: [batch_size * 2, seq_len]
        except RuntimeError:
            print("PADDING NOW")
            generated_tokens = pad_sequence(
                generated_tokens, batch_first=True, padding_value=_pad_token
            )
            generated_masks = pad_sequence(
                generated_masks, batch_first=True, padding_value=0
            )

        metrics = {
            f"{prefix}/mean_energy": energies.mean(),
            f"{prefix}/std_energy": energies.std(),
            f"{prefix}/max_energy": energies.max(),
            f"{prefix}/min_energy": energies.min(),
        }

        # Log raw reward statistics if reward shaping is enabled
        if self.use_reward_shaping:
            metrics.update(
                {
                    f"{prefix}/mean_raw_reward": raw_rewards_tensor.mean(),
                    f"{prefix}/std_raw_reward": raw_rewards_tensor.std(),
                    f"{prefix}/max_raw_reward": raw_rewards_tensor.max(),
                    f"{prefix}/min_raw_reward": raw_rewards_tensor.min(),
                }
            )

        # Compute policy logps using current model
        policy_logps = self._compute_policy_logps(
            self.model, generated_tokens, None, generated_masks
        )

        # Ignore the BOS when computing logps
        policy_logps = policy_logps * generated_masks[:, 1:]
        policy_logps = policy_logps.sum(-1)

        # Apply same beta and gamma transformations as original ERA
        beta_prime = self.alignment_betas / (
            1 + self.hparams.training_args.lightning_model_args.gamma
        )
        gamma_prime = self.hparams.training_args.lightning_model_args.gamma / (
            1 + self.hparams.training_args.lightning_model_args.gamma
        )
        energies = energies * beta_prime

        # Compute reference policy probabilities for the same sequences
        with torch.no_grad():
            ref_logps = self._compute_policy_logps(
                self.reference_model, generated_tokens, None, generated_masks
            )

        # Ignore the BOS when computing logps
        ref_logps = ref_logps * generated_masks[:, 1:]
        ref_logps = ref_logps.sum(-1)

        # Compute KL loss using same formula as original ERA
        policy_logps_y1 = policy_logps.reshape(-1, 2)[:, 0]
        policy_logps_y2 = policy_logps.reshape(-1, 2)[:, 1]

        ref_logps_y1 = ref_logps.reshape(-1, 2)[:, 0]
        ref_logps_y2 = ref_logps.reshape(-1, 2)[:, 1]

        energies_y1 = energies.reshape(-1, 2)[:, 0]
        energies_y2 = energies.reshape(-1, 2)[:, 1]

        logp = nn.functional.logsigmoid(policy_logps_y2 - policy_logps_y1)
        logp_prime = nn.functional.logsigmoid(policy_logps_y1 - policy_logps_y2)

        logp_star = nn.functional.logsigmoid(
            -(energies_y2 - energies_y1) + (gamma_prime * (ref_logps_y2 - ref_logps_y1))
        )
        logp_star_prime = nn.functional.logsigmoid(
            -(energies_y1 - energies_y2) + (gamma_prime * (ref_logps_y1 - ref_logps_y2))
        )
        kl_loss = torch.exp(logp_star) * (logp_star - logp) + torch.exp(
            logp_star_prime
        ) * (logp_star_prime - logp_prime)

        # Compute probability ratio: (pi_theta(y) * pi_theta(y')) / (pi_ref(y) * pi_ref(y'))
        log_prob_ratio = (policy_logps_y1 + policy_logps_y2) - (
            ref_logps_y1 + ref_logps_y2
        )
        eps = self.hparams.training_args.lightning_model_args.get("eps", 1e-10)

        log_total_loss = torch.logsumexp(
            log_prob_ratio + (kl_loss + eps).log(), dim=0
        ) - torch.log(torch.tensor(kl_loss.shape[0], device=self.device))

        total_loss = torch.exp(log_total_loss)

        log_ratio_shifted = log_prob_ratio - log_prob_ratio.max()
        log_sum_weights = torch.logsumexp(log_ratio_shifted, dim=0)
        log_sum_weights_squared = torch.logsumexp(2 * log_ratio_shifted, dim=0)
        log_ess = 2 * log_sum_weights - log_sum_weights_squared
        effective_sample_size = torch.exp(log_ess) / kl_loss.shape[0]

        metrics[f"{prefix}/ERAOnlineLoss"] = total_loss
        metrics[f"{prefix}/EffectiveSampleSize"] = effective_sample_size

        # Update reference model periodically (default every 100 steps)
        update_ess_threshold = self.hparams.training_args.lightning_model_args.get(
            "reference_update_ess_threshold", None
        )

        if update_ess_threshold is not None:
            if effective_sample_size <= update_ess_threshold:
                self.reference_model.load_state_dict(self.model.state_dict())
                print(
                    f"Step {self.global_step}: Updated reference model (ESS: {effective_sample_size:.4f} <= {update_ess_threshold})"
                )

        self.log_dict(
            metrics,
            on_epoch=True,
            on_step=self.hparams.training_args.lightning_model_args.on_step,
            sync_dist=self.hparams.training_args.lightning_model_args.sync_dist,
            batch_size=tokens.shape[0],
        )

        return total_loss

    def _shared_eval_grpo(self, batch, batch_idx, prefix):
        """
        Group relative policy optimization (GRPO) evaluation with DAPO token-level loss.
        (https://huggingface.co/papers/2503.14476)

        Differences from original GRPO:
        - Uses token-level normalization: L = -1/N * sum(per_token_loss * completion_mask)
        - N is the total number of completion tokens across the entire batch
        - Eliminates response-length bias where longer responses are under-penalized
        - Provides fair weighting regardless of individual response lengths

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
            # Controls the width of the entropy reward curve. Higher values allow more variability.
            # Default 1.0 is tight; increase to 2.0, 3.0 etc for wider tolerance around target entropy
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
                    "protgps_parent_dir", "/home/jxliu2/protgps"
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

        ########## ORIGINAL GRPO

        # # Sum loss across group elements to yield per_prompt_loss

        # per_prompt_loss_dict = {}
        # for prompt_idx, loss in group_losses.items():
        #     per_prompt_loss = torch.stack(loss).sum(dim=0)
        #     per_prompt_loss_dict[prompt_idx] = per_prompt_loss

        # # Sum total number of response tokens for each prompt_idx
        # per_prompt_token_count_dict = {}
        # for i, seq_data in enumerate(generated_sequences):
        #     prompt_idx = seq_data['prompt_idx']
        #     response_mask = seq_data['response_mask']
        #     token_count = response_mask.sum()
        #     if prompt_idx not in per_prompt_token_count_dict:
        #         per_prompt_token_count_dict[prompt_idx] = torch.tensor(0.0, device=self.device)
        #     per_prompt_token_count_dict[prompt_idx] += token_count

        # # Normalize per_prompt_loss_dict by total number of response tokens
        # for prompt_idx, loss in per_prompt_loss_dict.items():
        #     per_prompt_loss_dict[prompt_idx] = per_prompt_loss_dict[prompt_idx] / per_prompt_token_count_dict[prompt_idx]

        # grpo_batch_loss = 0
        # for prompt_idx, loss in per_prompt_loss_dict.items():
        #     grpo_batch_loss += loss
        # grpo_batch_loss = grpo_batch_loss / batch_size

        ########## DAPO
        # Reference: https://huggingface.co/papers/2503.14476
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
        total_entropy_penalty_sum = 0.0
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

    def _update_reference_model(self):
        """Update the reference model with current model parameters"""
        if (
            self.training_args["training_mode"] == "era_online"
            and self.reference_model is not None
        ):
            self.reference_model.load_state_dict(self.model.state_dict())
            # Ensure reference model stays frozen and in eval mode
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()

    def _shared_eval_dpo(self, batch, batch_idx, prefix):
        if not hasattr(self, "alignment_betas"):
            self.alignment_betas = torch.tensor(
                self.hparams.training_args.lightning_model_args.beta
            ).to(self.device)

        tokens, masks, energies, ref_logps = batch

        policy_logps = self._compute_policy_logps(self.model, tokens, None, masks)
        # Ignore the BOS when computing logps
        policy_logps = policy_logps * masks[:, 1:]
        policy_logps = policy_logps.sum(-1)

        beta = self.alignment_betas

        if energies.dim() == 2:
            energies = energies.sum(-1)

        policy_logps_y1 = policy_logps.reshape(-1, 2)[:, 0]
        policy_logps_y2 = policy_logps.reshape(-1, 2)[:, 1]

        ref_logps_y1 = ref_logps.reshape(-1, 2)[:, 0]
        ref_logps_y2 = ref_logps.reshape(-1, 2)[:, 1]

        energies_y1 = energies.reshape(-1, 2)[:, 0]
        energies_y2 = energies.reshape(-1, 2)[:, 1]

        # SI: want to be able to specify whether higher or lower energy is better
        if self.hparams.training_args.lightning_model_args.better_energy == "higher":
            y2_sign = (energies_y2 >= energies_y1).long()
        elif self.hparams.training_args.lightning_model_args.better_energy == "lower":
            y2_sign = (energies_y2 <= energies_y1).long()
        else:
            raise ValueError("better_energy must be 'higher' or 'lower'")
        y2_sign[y2_sign == 0] = -1
        y1_sign = -y2_sign

        pi_logratios = y2_sign * policy_logps_y2 + y1_sign * policy_logps_y1
        ref_logratios = y2_sign * ref_logps_y2 + y1_sign * ref_logps_y1

        loss = -nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios))
        tot_loss = loss.mean()

        metrics = {f"{prefix}/DPOLoss": tot_loss}

        self.log_dict(
            metrics,
            on_epoch=True,
            on_step=self.hparams.training_args.lightning_model_args.on_step,
            sync_dist=self.hparams.training_args.lightning_model_args.sync_dist,
            batch_size=tokens.shape[0],
        )

        return tot_loss

    def _get_unmask_indices(self, masked_indices, sequence_length):
        num_masked = masked_indices.shape[0]
        if num_masked == 0:
            return np.array([])

        if np.random.rand() < 0.8:
            # Draw from Beta(3, 9) distribution
            mask_rate = np.random.beta(3, 9)
        else:
            # Draw from a linear distribution
            mask_rate = np.random.uniform(0, 1)

        num_to_mask = int(mask_rate * sequence_length)
        num_to_mask = max(1, num_to_mask)
        num_to_mask = min(num_to_mask, num_masked)
        mask_indices = np.random.choice(masked_indices, num_to_mask, replace=False)
        return mask_indices

    def _unmask_all_indices(self, masked_indices, sequence_length):
        num_masked = masked_indices.shape[0]
        if num_masked == 0:
            return np.array([])
        # Return all the indices at once for unmasking everything
        return masked_indices

    def on_before_optimizer_step(self, optimizer):
        # Calculate the L2 norm of the gradients
        norms = grad_norm(self, norm_type=2)
        # Log the norms, for example, to TensorBoard
        self.log_dict(norms, prog_bar=False)

    def training_step(self, batch, batch_idx):
        if self.training_args["training_mode"] == "autoregressive":
            return self._shared_eval_autoreg(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "mixed_sequence":
            return self._shared_eval_mixed_seq(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "bidirectional":
            return self._shared_eval_bidirec(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "transfusion":
            return self._shared_eval_transfusion(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "transfusion_offset":
            return self._shared_eval_transfusion_offset(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "era":
            return self._shared_eval_era(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "era_online":
            return self._shared_eval_era_online(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "grpo":
            return self._shared_eval_grpo(batch, batch_idx, "train")
        elif self.training_args["training_mode"] == "dpo":
            return self._shared_eval_dpo(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            if self.training_args["training_mode"] == "autoregressive":
                return self._shared_eval_autoreg(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "mixed_sequence":
                return self._shared_eval_mixed_seq(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "bidirectional":
                return self._shared_eval_bidirec(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "transfusion":
                return self._shared_eval_transfusion(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "transfusion_offset":
                return self._shared_eval_transfusion_offset(
                    batch, batch_idx, "validation"
                )
            elif self.training_args["training_mode"] == "era":
                return self._shared_eval_era(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "era_online":
                return self._shared_eval_era_online(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "grpo":
                return self._shared_eval_grpo(batch, batch_idx, "validation")
            elif self.training_args["training_mode"] == "dpo":
                return self._shared_eval_dpo(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            if self.training_args["training_mode"] == "autoregressive":
                return self._shared_eval_autoreg(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "mixed_sequence":
                return self._shared_eval_mixed_seq(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "bidirectional":
                return self._shared_eval_bidirec(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "transfusion":
                return self._shared_eval_transfusion(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "transfusion_offset":
                return self._shared_eval_transfusion_offset(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "era":
                return self._shared_eval_era(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "era_online":
                return self._shared_eval_era_online(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "grpo":
                return self._shared_eval_grpo(batch, batch_idx, "test")
            elif self.training_args["training_mode"] == "dpo":
                return self._shared_eval_dpo(batch, batch_idx, "test")

    def forward_bidirec(self, batch, token_sampler, unmasking_mode="sample"):
        """Does not currently add padding"""
        if unmasking_mode == "sample" or unmasking_mode == "all":
            struct_batch, _, masked_smi_batch, seq_id, _ = batch
            device = struct_batch.device

            batch_size = masked_smi_batch.shape[0]
            padded_sequence_length = masked_smi_batch.shape[1]

            # Fix this hardcoding
            mask_token = self.token_info["input"]["TOK"]["TOK_MASK"]
            pad_token = self.token_info["input"]["TOK"]["TOK_PAD"]

            smiles_masked_indices = (
                masked_smi_batch == mask_token
            )  # GET THIS, remove hardcoding after testing
            # True sequence length without padding and masking
            # rotamer_non_special_indices = ((masked_rotamer_tokens_batch <= self.max_non_special_token_rotamer)
            #                                | (masked_rotamer_tokens_batch == self.mask_token_rotamer))
            # sequence_length = rotamer_non_special_indices.sum(-1)
            sequence_length = (
                (masked_smi_batch < pad_token) | (masked_smi_batch == mask_token)
            ).sum(-1)

            masked_indices = [
                torch.where(smiles_masked_indices[i])[0].detach().cpu().numpy()
                for i in range(batch_size)
            ]

            if unmasking_mode == "sample":
                unmasked_indices = [
                    (
                        self._get_unmask_indices(masked_indices[i], sequence_length[i])
                        + padded_sequence_length * i
                    )
                    for i in range(batch_size)
                ]
            elif unmasking_mode == "all" or unmasking_mode == "masked_left_to_right":
                unmasked_indices = [
                    (
                        self._unmask_all_indices(masked_indices[i], sequence_length[i])
                        + padded_sequence_length * i
                    )
                    for i in range(batch_size)
                ]

            unmasked_indices = torch.tensor(
                np.concatenate(unmasked_indices), device=device
            ).long()

            # import pdb; pdb.set_trace()
            while unmasked_indices.shape[0] > 0:
                # logits = self.model(structural_tokens=structural_tokens_batch,
                #                  residue_tokens=residue_tokens_batch,
                #                  rotamer_tokens=masked_rotamer_tokens_batch,
                #                  bb_coords=bb_coords_batch,
                #                  sequence_id=sequence_id_batch)
                # import pdb; pdb.set_trace()
                logits = self.model(masked_smi_batch, struct_batch, seq_id)

                masked_smi_batch_flattened = masked_smi_batch.flatten()
                logits_flattened = logits.view(-1, logits.shape[-1])

                # mask_token_probs = nn.functional.softmax(
                #     logits_flattened[unmasked_indices], dim=-1
                # )
                # tokens = torch.multinomial(mask_token_probs, 1).squeeze(-1)
                tokens, _ = token_sampler(logits_flattened[unmasked_indices]).squeeze(
                    -1
                )

                masked_smi_batch_flattened[unmasked_indices] = tokens

                masked_smi_batch = masked_smi_batch_flattened.view(
                    -1, masked_smi_batch.shape[-1]
                )

                smi_masked_indices = masked_smi_batch == mask_token

                masked_indices = [
                    torch.where(smi_masked_indices[i])[0].detach().cpu().numpy()
                    for i in range(batch_size)
                ]
                unmasked_indices = [
                    (
                        self._get_unmask_indices(masked_indices[i], sequence_length[i])
                        + padded_sequence_length * i
                    )
                    for i in range(batch_size)
                ]

                unmasked_indices = torch.tensor(
                    np.concatenate(unmasked_indices), device=device
                ).long()

            return masked_smi_batch

        elif unmasking_mode == "masked_left_to_right":
            structural_batch, smi_batch, seq_id_batch = batch
            assert (
                structural_batch.shape[0] == smi_batch.shape[0] == seq_id_batch.shape[0]
            )
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

            # Fix this hardcoding
            mask_token = self.token_info["input"]["TOK"]["TOK_MASK"]
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

                # SI: check where there's a mask token at the end of each sequence still being worked on and remove it
                if working_smi_batch[:, -1].eq(mask_token).all():
                    working_smi_batch = working_smi_batch[:, :-1]
                    working_token_probs = working_token_probs[:, :-1]
                    working_seq_id_batch = working_seq_id_batch[:, :-1]
                    working_structural_batch = working_structural_batch[:, :-1]

                logits = self.model(
                    working_smi_batch, working_structural_batch, working_seq_id_batch
                )  # (N, T, E)
                next_pos = logits[:, -1, :]  # (N, E)
                # char_probs = torch.nn.functional.softmax(next_pos, dim=-1)  # (N, E)
                # tokens = torch.multinomial(char_probs, 1)  # (N, 1)
                # selected_probs = torch.gather(char_probs, 1, tokens)  # (N, 1)
                tokens, selected_probs = token_sampler(next_pos)

                # import pdb; pdb.set_trace()
                concatenated_results = torch.cat((working_smi_batch, tokens), dim=-1)
                concatenated_probs = torch.cat(
                    (working_token_probs, selected_probs), dim=1
                )

                # SI: Append mask token to end of sequence
                mask_tokens = torch.full(
                    (working_smi_batch.shape[0], 1), mask_token, device=device
                )
                concatenated_results = torch.cat(
                    (concatenated_results, mask_tokens), dim=-1
                )

                # SI: Append mask token probability (zero as a filler) to end of sequence
                mask_probs = torch.zeros((working_smi_batch.shape[0], 1), device=device)
                concatenated_probs = torch.cat((concatenated_probs, mask_probs), dim=-1)

                # Sequence id here is binary so appending on more 1's is good enough
                # SI: needed to append two at a time to account for presence of mask token
                concatenated_seq_id = torch.cat(
                    (
                        working_seq_id_batch,
                        torch.ones(
                            (working_seq_id_batch.shape[0], 2), device=device
                        ).long(),
                    ),
                    dim=1,
                )
                concatenated_structural = torch.cat(
                    (
                        working_structural_batch,
                        torch.ones(
                            (working_structural_batch.shape[0], 2), device=device
                        ).long()
                        * pad_token,
                    ),
                    dim=1,
                )

                stop_token_mask = (
                    concatenated_results[:, -2] == stop_token
                )  # modified from original autorereg code
                comp_structs = concatenated_results[stop_token_mask]
                comp_probs = concatenated_probs[stop_token_mask]
                comp_inds = index_mapping[stop_token_mask]

                # SI: remove the last token (mask token) from the completed structures and probabilities
                comp_structs = comp_structs[:, :-1]
                comp_probs = comp_probs[:, :-1]

                for i, icomp in enumerate(comp_inds):
                    completed_structures[icomp] = comp_structs[i].detach().cpu().numpy()
                    completed_token_probs[icomp] = comp_probs[i].detach().cpu().numpy()

                # import pdb; pdb.set_trace()
                working_smi_batch = concatenated_results[~stop_token_mask]
                working_token_probs = concatenated_probs[~stop_token_mask]
                index_mapping = index_mapping[~stop_token_mask]
                # Also update the sequence ID and structural tokens
                working_seq_id_batch = concatenated_seq_id[~stop_token_mask]
                working_structural_batch = concatenated_structural[~stop_token_mask]

                if (
                    working_smi_batch.shape[-1] > 1000
                ):  # Fix this hardcoding for the token limit
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

    def _sample_structue_tokens_unmasking(
        self,
        completed_smiles_structures,
        token_sampler,
        smiles_pad_token,
        struct_token,
        struct_mask_token,
        struct_stop_token,
        device,
    ):
        """
        With completed SMILES structures, sample the structure tokens
        """
        # import pdb; pdb.set_trace()
        batch_size = len(completed_smiles_structures)
        # Stage 2: Structural inference
        final_completed_smiles = []
        final_completed_structure = []
        # Create a batch from the ragged tensors but augment by a buffer value of doubling the size
        working_smiles_batch = pad_sequence(
            completed_smiles_structures,
            batch_first=True,
            padding_value=smiles_pad_token,
        )  # (N, T)
        working_smiles_batch = torch.cat(
            (
                working_smiles_batch,
                torch.ones_like(working_smiles_batch) * smiles_pad_token,
            ),
            dim=1,
        )  # (N, 2*T)
        working_seq_id_batch = (
            working_smiles_batch != smiles_pad_token
        ).long()  # (N, 2*T)
        # Add a structure token to the end of the batch using the sum
        working_smiles_batch[
            torch.arange(working_smiles_batch.shape[0]), working_seq_id_batch.sum(-1)
        ] = struct_token
        working_seq_id_batch = (
            working_smiles_batch != smiles_pad_token
        ).long()  # (N, 2*T)
        working_structural_batch = (
            torch.tensor([struct_mask_token] * batch_size).reshape(-1, 1).to(device)
        )  # (N, 1)

        while len(final_completed_smiles) < batch_size:
            curr_batch = (
                "smiles_and_struct",
                working_smiles_batch,
                working_structural_batch,
                working_seq_id_batch,
            )
            out = self.model(curr_batch)
            _, struct_out, _, _ = (
                out  # struct_out shape is (N*S, E), need to get back to (N, S, E)
            )

            # Convert the probabilities to tokens
            input_batch_size = working_structural_batch.shape[0]
            struct_out = struct_out.reshape(
                input_batch_size, -1, struct_out.shape[-1]
            )  # (N, S, E)
            struct_next_pos = struct_out[:, -1, :]  # (N, E)
            # struct_char_probs = torch.nn.functional.softmax(
            #     struct_next_pos, dim=-1
            # )  # (N, E)
            # struct_tokens = torch.multinomial(struct_char_probs, 1)
            struct_tokens, _ = token_sampler(struct_next_pos)
            struct_tokens = struct_tokens.squeeze(
                -1
            )  # (N,) to allow for direct assignment

            # Filter out completed stuff
            working_structural_batch[:, -1] = struct_tokens
            stop_token_mask = working_structural_batch[:, -1] == struct_stop_token

            final_completed_smiles.extend([*working_smiles_batch[stop_token_mask]])
            final_completed_structure.extend(
                [*working_structural_batch[stop_token_mask]]
            )

            working_structural_batch = working_structural_batch[~stop_token_mask]
            working_smiles_batch = working_smiles_batch[~stop_token_mask]
            working_seq_id_batch = working_seq_id_batch[~stop_token_mask]

            # Update the modified quantities

            # Check if smiles batch needs resizing
            last_tokens = working_smiles_batch[:, -1]
            if (last_tokens != smiles_pad_token).any():
                # Add a buffering
                working_smiles_batch = torch.cat(
                    (
                        working_smiles_batch,
                        torch.ones_like(working_smiles_batch) * smiles_pad_token,
                    ),
                    dim=1,
                )
                working_seq_id_batch = (working_smiles_batch != smiles_pad_token).long()
            # Add a structure token to the end of the batch using the sum
            working_smiles_batch[
                torch.arange(working_smiles_batch.shape[0]),
                working_seq_id_batch.sum(-1),
            ] = struct_token
            working_seq_id_batch = (working_smiles_batch != smiles_pad_token).long()

            # Append a structure masking token to the end of the structural batch
            working_structural_batch = torch.cat(
                (
                    working_structural_batch,
                    torch.tensor(
                        [struct_mask_token] * working_structural_batch.shape[0]
                    )
                    .reshape(-1, 1)
                    .to(device),
                ),
                dim=-1,
            )

            assert (
                (working_smiles_batch == struct_token)
                .sum(-1)
                .eq(working_structural_batch.shape[-1])
                .all()
            )

            if working_structural_batch.shape[-1] > 1000:
                # FH: Introduce a hard cutoff to prevent infinite looping
                final_completed_smiles.extend([*working_smiles_batch])
                final_completed_structure.extend([*working_structural_batch])

        # Detach and cast to cpu
        final_completed_smiles = [
            x.detach().cpu().numpy() for x in final_completed_smiles
        ]
        final_completed_structure = [
            x.detach().cpu().numpy() for x in final_completed_structure
        ]

        return final_completed_smiles, final_completed_structure

    def _determine_maximum_num_struct_tokens(
        self, completed_smiles_structures, alphabet
    ):
        # The tightest possible upper bound for a batch is the greatest number of atoms
        #   for a sequence in the batch
        needed_lengths = []
        for seq in completed_smiles_structures:
            non_special_tokens = (
                seq[1:-1].detach().cpu().numpy()
            )  # Remove start, stop tokens
            temp_smi = "".join(alphabet[non_special_tokens])
            try:
                mol = Chem.MolFromSmiles(temp_smi)
                mol = Chem.AddHs(mol)
                num_atoms = mol.GetNumAtoms()
                needed_lengths.append(num_atoms)
            except Exception:
                # Just a single structure token
                needed_lengths.append(1)
        needed_lengths = torch.tensor(
            needed_lengths,
            device=completed_smiles_structures[0].device,
            dtype=torch.long,
        )
        return needed_lengths

    def _get_node_features(self, completed_smiles_structures, alphabet):
        """
        Extracts node features after removing all the hydrogen atoms from the
        molecule
        """
        # import pdb; pdb.set_trace()
        mol_node_features = []
        failed_indices = []
        for i, seq in enumerate(completed_smiles_structures):
            non_special_tokens = seq[1:-1].detach().cpu().numpy()
            temp_smi = "".join(alphabet[non_special_tokens])
            try:
                mol = Chem.MolFromSmiles(temp_smi)
                # The transformer only sees heavy atoms, no protons!
                mol = Chem.RemoveAllHs(mol)
                node_features = np.array(get_node_features(mol))
                mol_node_features.append(node_features)
            except Exception:
                # import pdb; pdb.set_trace()
                # Just a single structure token
                failed_indices.append(i)
                mol_node_features.append([])
        return mol_node_features, failed_indices

    def _generate_token_group_masking(
        self, mol_node_features, token_grouping, total_vocabulary_size
    ):
        """
        The mol_node_features contain the atom-valency-hybridization information
        for each atom in the molecule. We know that only certain tokens are allowed so we need to mask
        out the rest of the vocabulary. The token_grouping dictionary maps keys of the form
        'atom-valency-hybridization' to the set of valid indices in the vocabulary.

        Note that because the probabilities are computed using softmax, to properly zero out the logits for
        undesirable tokens, we need to set them to -inf. We do this by using the addition identity, and only
        adding 0 to logits which we want to keep (any values + -inf = -inf).

        mol_node_features: (n_atoms, 3)
        token_grouping: {atom-valency-hybridization: [indices]}
        total_vocabulary_size: int
        """
        masking = torch.full(
            (len(mol_node_features), total_vocabulary_size),
            float("-inf"),
            dtype=torch.float32,
        )
        for i, node_features in enumerate(mol_node_features):
            atom_val_hyb = "_".join([str(int(x)) for x in node_features])
            if atom_val_hyb in token_grouping:
                valid_indices = token_grouping[atom_val_hyb]
                masking[i, valid_indices] = 0.0
            else:
                # All tokens are viable here
                masking[i, :] = 0.0
        return masking

    def _construct_packed_sequence_repr(
        self, mol_node_features, avh_special_tokens, structure_masking_token
    ):
        """
        Construct the packed sequences with the correct offsets, simlar to the output of the
        input generator where a final sequence looks like:
        [ P, S0,   P, S1,  P, S2,  P, S3]
        [A0,  P,  A1,  P, A2,  P, A3,  P]
        [V0,  P,  V1,  P, V2,  P, V3,  P]
        [H0,  P,  H1,  P, H2,  P, H3,  P]

        See "idr_plm.nn.transformer.input_generators.SMILESStructureAVHGenerator" for more information.
        Here, mol_node_features is (n_atom, 3)
        """
        mol_node_features = torch.tensor(
            mol_node_features, dtype=torch.long
        )  # (n_atoms, 3)
        n_structure_tokens = len(mol_node_features)
        completed_repr = torch.zeros((4, n_structure_tokens * 2), dtype=torch.long)
        node_feat_T = mol_node_features.T  # (3, n_atoms)
        completed_repr[0, :] = structure_masking_token
        completed_repr[1, 0::2] = node_feat_T[0, :]  # Atom
        completed_repr[1, 1::2] = avh_special_tokens["atom_pad_idx"]
        completed_repr[2, 0::2] = node_feat_T[1, :]
        completed_repr[2, 1::2] = avh_special_tokens["valency_pad_idx"]
        completed_repr[3, 0::2] = node_feat_T[2, :]
        completed_repr[3, 1::2] = avh_special_tokens["hybrid_pad_idx"]
        return completed_repr

    def _sample_structure_tokens_autorgressive_avh(
        self,
        all_smiles_reprs,
        all_node_reprs,
        all_seq_id,
        all_vocab_masks,
        all_num_iterations,
        smiles_pad_token,
        struct_pad_token,
    ):
        """
        Sample the structure tokens autoregressively using the AVH information,
        but updating the structure track along the way with the tokens that are sampled
        """
        # import pdb; pdb.set_trace()
        completed_smiles_tracks = []
        completed_structure_tracks = []
        max_iter = max(all_num_iterations)

        # Construct a batching vector for reconstruction of struct_out, tricky because of how
        #   all structure tokens are passed at once
        node_repr_T = all_node_reprs.permute(0, 2, 1)  # (N, S, 4)
        batch_vector = node_repr_T[:, :, 0] != struct_pad_token  # (N, S)
        cat_1 = (
            torch.ones(batch_vector.shape[0], dtype=torch.long)
            .reshape(-1, 1)
            .to(batch_vector.device)
        )  # (N, 1)
        batch_vector = torch.cat((cat_1, batch_vector), dim=-1)  # (N, S + 1)
        batch_vector = batch_vector.reshape(-1)

        for i in range(max_iter):
            input_batch_size = all_smiles_reprs.shape[0]
            curr_batch = (
                "smiles_struct_avh",
                all_smiles_reprs,
                all_node_reprs,
                all_seq_id,
            )
            out = self.model(curr_batch)
            # import pdb; pdb.set_trace()
            _, struct_out, _, _ = out
            # This reshaping does not work because you don't have the exact same number of structure tokens for each
            #     sequence in the batch
            # struct_out = struct_out.reshape(input_batch_size, -1, struct_out.shape[-1]) #(N, S, E)

            # Try a more costly strategy of reshaping the entire batch
            struct_out_tmp = torch.zeros(
                batch_vector.shape[0], struct_out.shape[-1]
            ).to(struct_out.device)
            struct_out_tmp[batch_vector.bool()] = struct_out
            struct_out = struct_out_tmp.reshape(
                input_batch_size, -1, struct_out.shape[-1]
            )  # (N, S, E)

            # Careful here: the index for the position we want to sample is actually the 2i + 1th position
            sample_idx = (2 * i) + 1
            struct_token_pos = struct_out[:, sample_idx, :]  # (N, E)
            # Print the max logits before and after masking
            # max_logits_pre_mask = torch.max(struct_token_pos, dim = -1)
            # print("Max logits before masking", max_logits_pre_mask)
            # Correct the struct_token_pos logits with the vocabulary masks
            struct_token_pos = struct_token_pos + all_vocab_masks[:, i, :]  # (N, E)
            # max_logits_post_mask = torch.max(struct_token_pos, dim = -1)
            # print("Max logits after masking", max_logits_post_mask)

            # Check the logit diff
            # logit_diff = max_logits_post_mask.values - max_logits_pre_mask.values
            # print("Logit diff", logit_diff)
            # import pdb; pdb.set_trace()
            struct_char_probs = torch.nn.functional.softmax(
                struct_token_pos, dim=-1
            )  # (N, E)
            struct_tokens = torch.multinomial(struct_char_probs, 1)  # (N, 1)
            struct_tokens = struct_tokens.squeeze(-1)  # (N,)
            # Update the working batch
            all_node_reprs[:, 0, sample_idx] = struct_tokens

            # Filter out those that are completed based on num_iterations
            # print("Before", all_node_reprs.shape)
            previous_shape = all_node_reprs.shape[0]

            num_sampled_tokens = i + 1

            num_sampled_mask = all_num_iterations == num_sampled_tokens
            completed_smiles_tracks.extend([*all_smiles_reprs[num_sampled_mask]])
            completed_structure_tracks.extend([*all_node_reprs[num_sampled_mask]])

            all_smiles_reprs = all_smiles_reprs[~num_sampled_mask]
            all_node_reprs = all_node_reprs[~num_sampled_mask]
            all_seq_id = all_seq_id[~num_sampled_mask]
            all_vocab_masks = all_vocab_masks[~num_sampled_mask]
            all_num_iterations = all_num_iterations[~num_sampled_mask]

            # print("After", all_node_reprs.shape)
            final_shape = all_node_reprs.shape[0]
            if previous_shape != final_shape:
                # Reconstruct the batch vector
                node_repr_T = all_node_reprs.permute(0, 2, 1)  # (N, S, 4)
                batch_vector = node_repr_T[:, :, 0] != struct_pad_token  # (N, S)
                cat_1 = (
                    torch.ones(batch_vector.shape[0], dtype=torch.long)
                    .reshape(-1, 1)
                    .to(batch_vector.device)
                )  # (N, 1)
                batch_vector = torch.cat((cat_1, batch_vector), dim=-1)  # (N, S + 1)
                batch_vector = batch_vector.reshape(-1)

        # Detach and cast to cpu
        completed_smiles_tracks = [
            x.detach().cpu().numpy() for x in completed_smiles_tracks
        ]
        completed_structure_tracks = [
            x.detach().cpu().numpy() for x in completed_structure_tracks
        ]
        return completed_smiles_tracks, completed_structure_tracks

    def _sample_structure_tokens_autoregressive(
        self,
        completed_smiles_structures,
        token_sampler,
        smiles_pad_token,
        struct_token,
        struct_mask_token,
        struct_stop_token,
        struct_pad_token,
        maximum_num_structure_tokens,
        device,
    ):
        """
        Sample structure tokens autoregressively.

        Because of the way that the autoregressive forward pass is set up for the model, we need to
        do a dummy forward pass to extract the first structure token

        There are two stopping conditions: either the model samples a stop token or it reaches the maximum
        number of structure tokens allowed for the sequence
        """
        batch_size = len(completed_smiles_structures)
        working_struct_token_max_number_mask = torch.clone(maximum_num_structure_tokens)
        final_completed_smiles = []
        final_completed_structure = []
        working_smiles_batch = pad_sequence(
            completed_smiles_structures,
            batch_first=True,
            padding_value=smiles_pad_token,
        )  # (N, T)
        working_smiles_batch = torch.cat(
            (
                working_smiles_batch,
                torch.ones_like(working_smiles_batch) * smiles_pad_token,
            ),
            dim=1,
        )  # (N, 2*T)
        working_seq_id_batch = (
            working_smiles_batch != smiles_pad_token
        ).long()  # (N, 2*T)
        # Add a single structure token to the end of the batch using the sum
        working_smiles_batch[
            torch.arange(working_smiles_batch.shape[0]), working_seq_id_batch.sum(-1)
        ] = struct_token
        working_seq_id_batch = (
            working_smiles_batch != smiles_pad_token
        ).long()  # (N, 2*T)
        # This is a dummy structure batch, will overwrite this after a dummy pass
        working_structural_batch = (
            torch.tensor([struct_mask_token] * batch_size).reshape(-1, 1).to(device)
        )  # (N, 1)

        # #FH: Determine the limit here for structure tokens based on the number of non-padding smiles tokens
        # #   and dcale by 2.5x as a buffer. Would be good to derive a tighter upper bound on this value
        # max_num_non_pad_smiles_tokens = (working_smiles_batch != smiles_pad_token).sum(-1).max().item()
        # max_num_struct_tokens = math.ceil(max_num_non_pad_smiles_tokens * 2.5)

        # Dummy pass to sample the first structure token
        curr_batch = (
            "smiles_and_struct",
            working_smiles_batch,
            working_structural_batch,
            working_seq_id_batch,
        )
        out = self.model(curr_batch)
        _, struct_out, _, _ = (
            out  # Struct out shape is (N*S, E), need to get back to (N, S, E)
        )
        struct_out = struct_out.reshape(
            batch_size, -1, struct_out.shape[-1]
        )  # (N, S, E)
        # The very first token is going to be the 0th token, should only be 2 tokens at this
        #    stage
        assert struct_out.shape[1] == 2
        struct_first_pos = struct_out[:, 0, :]  # (N, E)
        # struct_char_probs = torch.nn.functional.softmax(
        #     struct_first_pos, dim=-1
        # )  # (N, E)
        # working_structural_batch = torch.multinomial(struct_char_probs, 1)  # (N, 1)
        working_structural_batch, _ = token_sampler(struct_first_pos)  # (N, 1)

        # Now we can perform the rest of the loop
        while len(final_completed_smiles) < batch_size:
            curr_batch = (
                "smiles_and_struct",
                working_smiles_batch,
                working_structural_batch,
                working_seq_id_batch,
            )
            out = self.model(curr_batch)
            input_batch_size = working_structural_batch.shape[0]
            _, struct_out, _, _ = out
            struct_out = struct_out.reshape(
                input_batch_size, -1, struct_out.shape[-1]
            )  # (N, S, E)
            struct_next_pos = struct_out[:, -1, :]  # (N, E)
            # struct_char_probs = torch.nn.functional.softmax(
            #     struct_next_pos, dim=-1
            # )  # (N, E)
            # struct_tokens = torch.multinomial(struct_char_probs, 1)
            struct_tokens, _ = token_sampler(struct_next_pos)

            # Stop token completion check
            working_structural_batch = torch.cat(
                (working_structural_batch, struct_tokens), dim=-1
            )
            stop_token_mask = working_structural_batch[:, -1] == struct_stop_token

            final_completed_smiles.extend([*working_smiles_batch[stop_token_mask]])
            final_completed_structure.extend(
                [*working_structural_batch[stop_token_mask]]
            )

            working_structural_batch = working_structural_batch[~stop_token_mask]
            working_smiles_batch = working_smiles_batch[~stop_token_mask]
            working_seq_id_batch = working_seq_id_batch[~stop_token_mask]
            working_struct_token_max_number_mask = working_struct_token_max_number_mask[
                ~stop_token_mask
            ]

            # Control token mask
            # Remove any cases here immediately where the model samples a token greater than the maximum permitted
            control_token_mask = working_structural_batch[:, -1] >= struct_pad_token
            working_structural_batch = working_structural_batch[~control_token_mask]
            working_smiles_batch = working_smiles_batch[~control_token_mask]
            working_seq_id_batch = working_seq_id_batch[~control_token_mask]
            working_struct_token_max_number_mask = working_struct_token_max_number_mask[
                ~control_token_mask
            ]
            batch_size = batch_size - control_token_mask.sum().item()

            # Structural token number completion check
            # The working_struct_token_max_number_mask is the maximum number of structure tokens that should be
            #   sampled for each sequence, so anywhere that has more structure tokens than the maximum should be removed
            struct_token_number_mask = (
                working_struct_token_max_number_mask
                < working_structural_batch.shape[-1]
            )

            final_completed_smiles.extend(
                [*working_smiles_batch[struct_token_number_mask]]
            )
            final_completed_structure.extend(
                [*working_structural_batch[struct_token_number_mask]]
            )

            working_structural_batch = working_structural_batch[
                ~struct_token_number_mask
            ]
            working_smiles_batch = working_smiles_batch[~struct_token_number_mask]
            working_seq_id_batch = working_seq_id_batch[~struct_token_number_mask]
            working_struct_token_max_number_mask = working_struct_token_max_number_mask[
                ~struct_token_number_mask
            ]

            last_tokens = working_smiles_batch[:, -1]
            if (last_tokens != smiles_pad_token).any():
                # Add a buffering
                working_smiles_batch = torch.cat(
                    (
                        working_smiles_batch,
                        torch.ones_like(working_smiles_batch) * smiles_pad_token,
                    ),
                    dim=1,
                )
                working_seq_id_batch = (working_smiles_batch != smiles_pad_token).long()
            # Add a structure token to the end of the batch using the sum
            working_smiles_batch[
                torch.arange(working_smiles_batch.shape[0]),
                working_seq_id_batch.sum(-1),
            ] = struct_token
            working_seq_id_batch = (working_smiles_batch != smiles_pad_token).long()

            # The number of structural tokens in the smiles sequence should equal the number of structural tokens
            #   that have been sampled
            assert (
                (working_smiles_batch == struct_token)
                .sum(-1)
                .eq(working_structural_batch.shape[-1])
                .all()
            )
            assert (
                working_struct_token_max_number_mask
                >= working_structural_batch.shape[-1]
            ).all()

            # FH: Introduce a hard cutoff based on buffered number of non-padding smiles tokens to prevent infinite looping
            if working_structural_batch.shape[-1] > 1000:
                final_completed_smiles.extend([*working_smiles_batch])
                final_completed_structure.extend([*working_structural_batch])

        # Detach and cast to cpu
        final_completed_smiles = [
            x.detach().cpu().numpy() for x in final_completed_smiles
        ]
        final_completed_structure = [
            x.detach().cpu().numpy() for x in final_completed_structure
        ]

        return final_completed_smiles, final_completed_structure

    def forward_seq_struct_mixed(
        self,
        batch,
        token_sampler,
        alphabet,
        mode="masked",
        struct_limit_method="n_atoms",
        struct_limit_multiplier=2.5,
        use_finished_smiles=False,
        avh_args=None,
    ):
        """
        This function supports autoregressive inference across the smiles and structure tokens
        as well as left-to-right unmasking of the tokens.

        Notation is:
            N: batch size
            T/S: sequence length
            E: embedding dimension

        The struct_limit_method is how to determine the correct number of structure tokens to sample, either
        based on the number of atoms in the molecule (n_atoms) or a multiple of the number of non-padding tokens
        (token_mult). In the case of struct_limit_method=='token_mult', the struct_limit_multiplier determines
        the upper bound as the number of non-padding SMILES tokens multiplied by the multiplier.

        Some updated functionalities:
            use_finished_smiles: If this is set to True, the function assumes that the
            avh_args: A dictionary that contains arguments required for the AVH-enhanced sampling.
        """
        # import pdb; pdb.set_trace()
        assert mode in ["masked", "autoregressive", "autoregressive_avh"]
        assert struct_limit_method in ["n_atoms", "token_mult"]
        assert struct_limit_multiplier >= 1

        # smiles_start_token = self.token_info['input']['TOK']['TOK_START']
        smiles_mask_token = self.token_info["input"]["TOK"]["TOK_MASK"]
        smiles_stop_token = self.token_info["input"]["TOK"]["TOK_STOP"]
        smiles_pad_token = self.token_info["input"]["TOK"]["TOK_PAD"]
        struct_token = self.token_info["input"]["STRUCT"]["STRUCT"]
        struct_stop_token = self.token_info["input"]["STRUCT"]["STRUCT_STOP"]
        struct_mask_token = self.token_info["input"]["STRUCT"]["STRUCT_MASK"]
        struct_pad_token = self.token_info["input"]["STRUCT"]["STRUCT_PAD"]

        if use_finished_smiles:
            completed_smiles_structures = batch[0]
            device = completed_smiles_structures[0].device
        else:
            start_tokens, seq_id = batch
            assert start_tokens.shape[0] == seq_id.shape[0]
            device = start_tokens.device
            batch_size = start_tokens.shape[0]
            completed_smiles_structures = []
            # import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()

            # Stage 1: Smiles inference
            working_smi_batch = start_tokens.clone()  # (N, 1)
            working_seq_id_batch = seq_id.clone()  # (N, 1), all 1's

            if mode == "masked":
                # Attach a mask token and update sequence id at the end for unmasking
                working_smi_batch = torch.cat(
                    (
                        working_smi_batch,
                        torch.tensor([smiles_mask_token] * working_smi_batch.shape[0])
                        .reshape(-1, 1)
                        .to(device),
                    ),
                    dim=-1,
                )
                working_seq_id_batch = torch.cat(
                    (
                        working_seq_id_batch,
                        torch.ones(
                            (working_seq_id_batch.shape[0], 1), device=device
                        ).long(),
                    ),
                    dim=1,
                )

            while len(completed_smiles_structures) < batch_size:
                # print(working_smi_batch)
                # Get next token
                curr_batch = ("smiles_only", working_smi_batch, working_seq_id_batch)
                out = self.model(curr_batch)
                logits, _ = out
                next_pos = logits[:, -1, :]  # (N, E)
                # char_probs = torch.nn.functional.softmax(next_pos, dim=-1)  # (N, E)
                # tokens = torch.multinomial(char_probs, 1)  # (N, 1)
                tokens, _ = token_sampler(next_pos)  # (N, 1), sampler directly callable

                if mode == "masked":
                    tokens = tokens.squeeze(-1)  # (N,) to allow for direct assignment
                    working_smi_batch[:, -1] = tokens
                elif mode in ["autoregressive", "autoregressive_avh"]:
                    working_smi_batch = torch.cat((working_smi_batch, tokens), dim=-1)
                    working_seq_id_batch = torch.cat(
                        (
                            working_seq_id_batch,
                            torch.ones(
                                (working_seq_id_batch.shape[0], 1), device=device
                            ).long(),
                        ),
                        dim=1,
                    )

                stop_token_mask = working_smi_batch[:, -1] == smiles_stop_token

                completed_results = working_smi_batch[stop_token_mask]
                completed_smiles_structures.extend([*completed_results])

                working_smi_batch = working_smi_batch[~stop_token_mask]
                working_seq_id_batch = working_seq_id_batch[~stop_token_mask]

                # Additional check here as a control token should never be sampled. Any
                #   SMILES that have this feature should be removed immediately. When we do this removal, we
                #   need to adjust the target batch size that guards the while loop
                control_token_mask = working_smi_batch[:, -1] >= smiles_pad_token
                working_smi_batch = working_smi_batch[~control_token_mask]
                working_seq_id_batch = working_seq_id_batch[~control_token_mask]
                batch_size = batch_size - control_token_mask.sum().item()

                if mode == "masked":
                    working_smi_batch = torch.cat(
                        (
                            working_smi_batch,
                            torch.tensor(
                                [smiles_mask_token] * working_smi_batch.shape[0]
                            )
                            .reshape(-1, 1)
                            .to(device),
                        ),
                        dim=-1,
                    )
                    working_seq_id_batch = torch.cat(
                        (
                            working_seq_id_batch,
                            torch.ones(
                                (working_seq_id_batch.shape[0], 1), device=device
                            ).long(),
                        ),
                        dim=1,
                    )

                if working_smi_batch.shape[-1] > 1000:
                    # FH: Introduce a hard cutoff to prevent infinite looping
                    completed_smiles_structures.extend([*working_smi_batch])

        if mode == "masked":
            final_completed_smiles, final_completed_structure = (
                self._sample_structue_tokens_unmasking(
                    completed_smiles_structures,
                    token_sampler,
                    smiles_pad_token,
                    struct_token,
                    struct_mask_token,
                    struct_stop_token,
                    device,
                )
            )
        elif mode == "autoregressive":
            if struct_limit_method == "n_atoms":
                maximum_num_structure_tokens = (
                    self._determine_maximum_num_struct_tokens(
                        completed_smiles_structures, alphabet
                    )
                )
            elif struct_limit_method == "token_mult":
                max_num_tokens = max(len(x) for x in completed_smiles_structures)
                max_num_tokens = math.ceil(max_num_tokens * struct_limit_multiplier)
                maximum_num_structure_tokens = torch.tensor(
                    [max_num_tokens] * len(completed_smiles_structures),
                    device=device,
                    dtype=torch.long,
                )
            final_completed_smiles, final_completed_structure = (
                self._sample_structure_tokens_autoregressive(
                    completed_smiles_structures,
                    token_sampler,
                    smiles_pad_token,
                    struct_token,
                    struct_mask_token,
                    struct_stop_token,
                    struct_pad_token,
                    maximum_num_structure_tokens,
                    device,
                )
            )

        elif mode == "autoregressive_avh":
            # import pdb; pdb.set_trace()
            assert avh_args is not None
            # Maps atomic numbers to a 0-indexed value for embedding
            assert "atom_mapping" in avh_args
            # Specifies for each atom-valency-hybridization combination the indices of the vocabulary
            assert "token_grouping" in avh_args
            # Specifies special tokens for the atom-valency-hybridization information such as the padding index
            #   for each separate track
            assert "avh_special_tokens" in avh_args
            # Specifies the total size of the vocabulary embedding for structure tokens
            assert "total_structure_vocab_size" in avh_args

            # FH: Autoregressive inference augmented with atom-valency-hybridization information on top, requires a different
            #   approach to ensure that the offset in the token sequences is correctly preserved

            # Step 1: Get the node features of the smiles structures. mol_node_features here is of shape (n_molecules, n_atoms, 3)
            mol_node_features, failed_indices = self._get_node_features(
                completed_smiles_structures, alphabet
            )
            failed_indices = set(failed_indices)  # Faster lookup
            assert len(mol_node_features) == len(completed_smiles_structures)
            completed_smiles_structures = [
                x
                for i, x in enumerate(completed_smiles_structures)
                if i not in failed_indices
            ]
            mol_node_features = [
                x for i, x in enumerate(mol_node_features) if i not in failed_indices
            ]

            # Step 2 + 3: For each set of node features, generate a corresponding set of masks over the entire vocabulary and
            #  remap the atom types in the node features to their 0-indexed values using the atom_mapping dictionary
            all_vocab_masks = []
            atom_mapping = avh_args["atom_mapping"]
            for i, node_features in enumerate(mol_node_features):
                vocabulary_masks = self._generate_token_group_masking(
                    node_features,
                    token_grouping=avh_args["token_grouping"],
                    total_vocabulary_size=avh_args["total_structure_vocab_size"],
                )
                all_vocab_masks.append(vocabulary_masks)
                elements = node_features[:, 0]
                node_features[:, 0] = [
                    atom_mapping[int(element)] for element in elements
                ]

            # import pdb; pdb.set_trace()

            # Step 4: Shape the SMILES and structure sequences into the correct complementary formats
            all_mol_node_reprs = []
            all_smiles_reprs = []
            all_num_iterations = []
            for i, node_features in enumerate(mol_node_features):
                num_structure_tokens = node_features.shape[0]
                curr_smiles_struct = completed_smiles_structures[i]
                # Need the full sequence here of 2 * num_structure tokens, BUT need to account for shifting of the sequence
                #   later on...
                curr_smiles_struct = torch.cat(
                    (
                        curr_smiles_struct,
                        torch.tensor([struct_token] * (num_structure_tokens * 2)).to(
                            device
                        ),
                    ),
                    dim=0,
                )  # (T + S)
                all_smiles_reprs.append(curr_smiles_struct)
                all_num_iterations.append(num_structure_tokens)

                node_repr = self._construct_packed_sequence_repr(
                    node_features,
                    avh_special_tokens=avh_args["avh_special_tokens"],
                    structure_masking_token=struct_mask_token,
                )
                # Transpose here for easier sequence padding later
                all_mol_node_reprs.append(node_repr.T)

            # import pdb; pdb.set_trace()
            all_smiles_reprs = pad_sequence(
                all_smiles_reprs, batch_first=True, padding_value=smiles_pad_token
            )  # (N, T)
            all_mol_node_reprs = pad_sequence(
                all_mol_node_reprs, batch_first=True, padding_value=struct_pad_token
            )  # (S, N)
            all_mol_node_reprs = all_mol_node_reprs.permute(0, 2, 1)
            all_mol_node_reprs = all_mol_node_reprs.to(device)  # (N, 4, S)
            all_mol_node_reprs[:, 1, :][
                all_mol_node_reprs[:, 1, :] == struct_pad_token
            ] = avh_args["avh_special_tokens"]["atom_pad_idx"]
            all_mol_node_reprs[:, 2, :][
                all_mol_node_reprs[:, 2, :] == struct_pad_token
            ] = avh_args["avh_special_tokens"]["valency_pad_idx"]
            all_mol_node_reprs[:, 3, :][
                all_mol_node_reprs[:, 3, :] == struct_pad_token
            ] = avh_args["avh_special_tokens"]["hybrid_pad_idx"]
            all_vocab_masks = pad_sequence(
                all_vocab_masks, batch_first=True, padding_value=0.0
            )
            all_vocab_masks = all_vocab_masks.to(device)
            all_num_iterations = torch.tensor(
                all_num_iterations, device=device, dtype=torch.long
            )  # (N,)
            # Also, get the sequence IDs here, just need to do this once
            all_seq_id_reprs = (all_smiles_reprs != smiles_pad_token).long()  # (N, T)

            # Finally, pass into the sampling function
            # Not going to apply the token sampler here because the vocabulary masks
            #   already restrict the tokens to a very small valid subset so no need to
            #   do probability filtering/sampling.
            final_completed_smiles, final_completed_structure = (
                self._sample_structure_tokens_autorgressive_avh(
                    all_smiles_reprs=all_smiles_reprs,
                    all_node_reprs=all_mol_node_reprs,
                    all_seq_id=all_seq_id_reprs,
                    all_vocab_masks=all_vocab_masks,
                    all_num_iterations=all_num_iterations,
                    smiles_pad_token=smiles_pad_token,
                    struct_pad_token=struct_pad_token,
                )
            )

        return final_completed_smiles, final_completed_structure

    def forward_autoregressive_prompted(
        self, batch, token_sampler, use_cache_here=False
    ):
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

        # import time
        # start_time = time.time()

        # Clear KV cache at the start of generation for this batch
        # self.clear_kv_cache()

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

                # if iteration == 2:
                #     logits_no_cache = self.model(
                #         working_smi_batch,
                #         working_structural_batch,
                #         working_seq_id_batch,
                #         batch_access_indices=all_indices,
                #         inference_iteration=self._inference_iteration,
                #         use_cache_here=False,
                #     )  # (B, T, E)

                #     # Check:
                #     # (logits_no_cache[:, 0:working_seq_id_batch.sum(-1)[0], :] - logits[:, 0:working_seq_id_batch.sum(-1)[0], :]).sum()
                #     # (logits_no_cache[:, 0:10, :] - logits[:, 0:10, :]).sum()

                # if iteration == 1:
                #     # Save logits to variable for later reference
                #     logits_iteration_1 = logits.clone()

                # if iteration == 2:
                #     # Save logits to variable and compare to iteration == 1 logits
                #     logits_iteration_2 = logits.clone()

                # assert torch.allclose(logits, logits_no_cache, atol=1e-5), f"Cached and non-cached logits differ by more than 1e-5: max diff = {(logits - logits_no_cache).abs().max().item()}"

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

    def _transfusion_autoregressive_step(
        self, smi_batch, seq_id_batch, token_sampler, device
    ):
        """Autoregressive sampling of SMILES strings using the transfusion model"""
        assert smi_batch.shape == seq_id_batch.shape

        # Stage 1: Autoregressively sample SMILES in batched mode until a struct_start_token is sampled
        working_smi_batch = smi_batch.clone()  # (N, 1)
        working_seq_id_batch = seq_id_batch.clone()  # (N, 1)
        batch_size = working_smi_batch.shape[0]
        # For keeping track and only working on incomplete structures
        index_mapping = torch.arange(batch_size, device=device)
        completed_structures = [None] * batch_size
        all_structures_completed = False

        struct_start_token = self.token_info["input"]["STRUCT"]["STRUCT_START"]

        while not all_structures_completed:
            # A TranfusionMolTransformer here
            out = self.model(
                working_smi_batch,
                None,  # No structural input
                None,  # No structural mask
                working_seq_id_batch,
                None,
            )  # No time step
            logits, _ = out  # (N, T, E)
            next_pos = logits[:, -1, :]  # (N, E)
            # char_probs = torch.nn.functional.softmax(next_pos, dim=-1)  # (N, E)
            # tokens = torch.multinomial(char_probs, 1)  # (N, 1)
            tokens, _ = token_sampler(next_pos)  # (N, 1), sampler directly callable

            concatenated_results = torch.cat((working_smi_batch, tokens), dim=-1)
            concatenated_seq_id = torch.cat(
                (
                    working_seq_id_batch,
                    torch.ones(
                        (working_seq_id_batch.shape[0], 1), device=device
                    ).long(),
                ),
                dim=1,
            )

            stop_mask = concatenated_results[:, -1] == struct_start_token
            comp_structs = concatenated_results[stop_mask]
            comp_inds = index_mapping[stop_mask]

            for i, icomp in enumerate(comp_inds):
                completed_structures[icomp] = comp_structs[i].detach().cpu().numpy()

            working_smi_batch = concatenated_results[~stop_mask]
            index_mapping = index_mapping[~stop_mask]
            working_seq_id_batch = concatenated_seq_id[~stop_mask]

            if (
                working_smi_batch.shape[-1] > 1000
            ):  # Fix this hardcoding for the token limit
                working_smi_batch = torch.cat(
                    (
                        working_smi_batch,
                        torch.tensor([struct_start_token] * working_smi_batch.shape[0])
                        .reshape(-1, 1)
                        .to(device),
                    ),
                    dim=-1,
                )
                for j, idx in enumerate(index_mapping):
                    completed_structures[idx] = (
                        working_smi_batch[j].detach().cpu().numpy()
                    )

                all_structures_completed = True

            if len(working_smi_batch) == 0:
                all_structures_completed = True

        return completed_structures

    def _transfusion_filtering_step(self, completed_structures, alphabet):
        """Removing invalid SMILES strings and computing dihedrals for valid ones"""
        valid_structures = []
        initial_num = len(completed_structures)
        for struct in completed_structures:
            try:
                # Remove the start token and structure start token here
                decoded_string = "".join(
                    [alphabet[int(token)] for token in struct[1:-1]]
                )
                mol = Chem.MolFromSmiles(decoded_string)
                assert mol is not None
                angles, indices = compute_dihedrals(decoded_string)
                assert len(angles) > 0
                valid_structures.append((struct, decoded_string, angles, indices))
            except Exception:
                continue  # Skip the structure if it is invalid
        return valid_structures, initial_num

    def _transfusion_diffusion_step(self, valid_structures, diff_config, device):
        """Carry out the torsional diffusion step for valid structures"""
        struct_token = self.token_info["input"]["STRUCT"]["STRUCT"]
        struct_end_token = self.token_info["input"]["STRUCT"]["STRUCT_END"]

        n_diff_samples = diff_config["n_samples"]
        n_time_steps = diff_config["n_time_steps"]
        feat_dim = diff_config["feat_dim"]
        sigma_min = diff_config["sigma_min"] * np.pi
        sigma_max = diff_config["sigma_max"] * np.pi

        final_predictions = []

        for i, (tokens, smiles, angles, indices) in enumerate(valid_structures):
            print(f"Starting on valid structure {i}")
            # Compose the input to the model
            tokens = torch.tensor(tokens).long()
            # Remove the struct_start_token from the sequence so that it matches the format of data
            #   used as input during training (?)
            # The structure start token is included now so should be okay to leave it in during
            #   inference
            # tokens = tokens[:-1]
            # import pdb; pdb.set_trace()
            num_angles = len(angles)
            sequence_id = torch.ones_like(tokens)
            sequence_id = torch.cat(
                (sequence_id, torch.ones(num_angles) * 2, torch.ones(1)), dim=0
            ).long()

            tokens = torch.cat(
                (
                    tokens,
                    torch.tensor([struct_token] * num_angles).long(),
                    torch.tensor([struct_end_token]).long(),
                ),
                dim=0,
            )
            assert tokens.shape == sequence_id.shape
            # Unsqueeze and reshape into num_samples
            tokens = tokens.unsqueeze(0).expand(
                n_diff_samples, -1
            )  # (n_diff_samples, T)
            sequence_id = sequence_id.unsqueeze(0).expand(
                n_diff_samples, -1
            )  # (n_diff_samples, T)

            # Construct a structure mask
            struct_mask = torch.ones(
                (n_diff_samples, feat_dim)
            )  # (n_diff_samples, feat_dim)
            struct_mask[:, :num_angles] = 0
            struct_mask = struct_mask.bool()  # (n_diff_samples, feat_dim)

            # Construct the input dictionary
            input_dict = {
                "token_input": tokens.to(device),
                "sequence_id": sequence_id.to(device),
                "struct_mask": struct_mask.to(device),
            }
            torsions = sample_torsions(
                self.model,
                input_dict,
                n_diff_samples,
                n_time_steps,
                feat_dim,
                sigma_min,
                sigma_max,
                device=device,
            )
            # import pdb; pdb.set_trace()
            torsions = torsions[
                :, :, :num_angles
            ]  # Only keep the angles that are non-padding
            final_predictions.append((smiles, angles, indices, torsions))

        return final_predictions

    def forward_transfusion(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, np.ndarray, int],
        token_sampler: TokenSampler,
        run_diffusion: bool = True,
    ) -> tuple[list, list]:
        """Forward pass for the transfusion model where tokens are sampled first autoregressively
        and then the structural components are sampled by reversing a diffusion process

        Args:
            batch: tuple[torch.Tensor, torch.Tensor, numpy.ndarray, int]
                A tuple which contains the following:
                    token_input: torch.Tensor
                        The token input tensor. Here, the token input is just the SMILES start token
                    sequence_id: torch.Tensor
                        The sequence id tensor which starts off as just a tensor of ones
                    alphabet: numpy.ndarray
                        The alphabet for decoding the token indices into SMILES strings
                    diffusion_config: dict
                        Dictionary with contains the following information:
                            n_samples: int
                                The number of samples to generate for each valid SMILES string sampled
                            feat_dim: int
                                The feature dimension for diffusion of the structural features
                            n_time_steps: int
                                The number of time steps for the diffusion process
                            sigma_min: float
                                The minimum value for the diffusion sigma relative to pi
                            sigma_max
            run_diffusion: bool
                A flag which indicates if the diffusion process should be included. If False, the returned
                predictions are just the completed SMILES strings from autoregressive sampling.

        Returns:
            Dihedral angles sampled from the model (if diffusion enabled) and the autoregressive SMILES strings

        Notes:
            The input to this method starts with only the tokens corresponding to the SMILES strings
            because we sample the smiles strings first and then append noised uniform vectors for the
            diffusion process. The diffusion process is then reversed to obtain the dihedral angles
        """
        smi_batch, seq_id_batch, alphabet, diff_config = batch
        device = smi_batch.device
        completed_structures = self._transfusion_autoregressive_step(
            smi_batch, seq_id_batch, token_sampler, device
        )
        if not run_diffusion:
            print("Skipping diffusion, returning only SMILES strings")
            return [], completed_structures
        valid_structures, initial_num = self._transfusion_filtering_step(
            completed_structures, alphabet
        )
        print("Number of initial structures", initial_num)
        print("Number of valid structures", len(valid_structures))

        if len(valid_structures) == 0:
            print("No valid SMILES were generated, cannot compute dihedrals/diffuse")
            return [], completed_structures

        final_predictions = self._transfusion_diffusion_step(
            valid_structures, diff_config, device
        )
        return final_predictions, completed_structures

    def forward_transfusion_structure_only(self, batch):
        """Forward where the model is only asked to sample dihedrals, not the SMILES strings themselves"""
        smiles_tokens, alphabet, diff_config = batch
        device = smiles_tokens[0].device
        smiles_tokens = [x.detach().cpu().numpy() for x in smiles_tokens]
        valid_structures, initial_num = self._transfusion_filtering_step(
            smiles_tokens, alphabet
        )
        print("Number of initial structures", initial_num)
        print("Number of valid structures", len(valid_structures))

        if len(valid_structures) == 0:
            print("No valid SMILES were generated, cannot compute dihedrals/diffuse")
            return []
        final_predictions = self._transfusion_diffusion_step(
            valid_structures, diff_config, device
        )
        return final_predictions, valid_structures

    def forward_masked_mixed_seq_infilling(self, batch, unmask_size=50):
        """
        Rather than doing left-to-right or sampling tokens to unmask, this method performs in-filling where
        the stop token is specified and the model is asked to fill in the mask tokens between the start and stop tokens.
        This applies for both the smiles and structural tokens.

        The models that should be used with this are the mixed sequence transformers trained WITHOUT always enforcing
        stop token masking. The downside is you would need to know the size of regions to unmask in advance
        """
        raise NotImplementedError("This method is not implemented yet")


####################################################################################################################
# EXPOSED SAMPLING METHODS
####################################################################################################################


def sample_components_from_bidirectional_transformer(
    transformer_model,
    structural_tokens,
    masked_smiles_tokens,
    sequence_id,
    token_sampler,
    inference_batch_size=128,
    unmasking_mode="sample",
):
    """Samples components from the transformer model (lightning module)"""
    transformer_model.model.eval()
    assert not transformer_model.model.training

    num_batches = (
        structural_tokens.shape[0] + inference_batch_size - 1
    ) // inference_batch_size
    if unmasking_mode == "sample":
        unmasked_rotamer_tokens = []
        for batch in range(num_batches):
            print(batch, num_batches)
            structural_tokens_batch = structural_tokens[
                batch * inference_batch_size : (batch + 1) * inference_batch_size
            ]
            masked_smiles_tokens_batch = masked_smiles_tokens[
                batch * inference_batch_size : (batch + 1) * inference_batch_size
            ]
            sequence_id_batch = sequence_id[
                batch * inference_batch_size : (batch + 1) * inference_batch_size
            ]

            unmasked_rotamer_tokens_batch = transformer_model.forward_bidirec(
                (
                    structural_tokens_batch,
                    None,
                    masked_smiles_tokens_batch,
                    sequence_id_batch,
                    None,  # No loss mask when infering components
                ),
                token_sampler=token_sampler,
                unmasking_mode=unmasking_mode,
            )
            unmasked_rotamer_tokens.append(unmasked_rotamer_tokens_batch)
        unmasked_rotamer_tokens = torch.cat(unmasked_rotamer_tokens, dim=0)
        return unmasked_rotamer_tokens

    elif unmasking_mode == "all":
        raise NotImplementedError("Unmasking mode all not implemented")

    elif unmasking_mode == "masked_left_to_right":
        all_sampled_tokens = []
        # all_token_probs = []

        for batch in range(num_batches):
            print(batch, num_batches)
            structural_tokens_batch = structural_tokens[
                batch * inference_batch_size : (batch + 1) * inference_batch_size
            ]
            smiles_tokens_batch = masked_smiles_tokens[
                batch * inference_batch_size : (batch + 1) * inference_batch_size
            ]
            sequence_id_batch = sequence_id[
                batch * inference_batch_size : (batch + 1) * inference_batch_size
            ]
            tokens, _ = transformer_model.forward_bidirec(
                (structural_tokens_batch, smiles_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
                unmasking_mode=unmasking_mode,
            )
            all_sampled_tokens.append(tokens)
            # all_token_probs.append(probs)

        all_sampled_tokens = torch.cat(all_sampled_tokens, dim=0)
        return all_sampled_tokens  # , all_token_probs


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
            print("Generating SMILES unprompted")
            tokens, probs = transformer_model.forward_autoregressive(
                (structural_tokens_batch, smiles_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        else:
            print("Generating SMILES prompted")
            tokens, probs = transformer_model.forward_autoregressive_prompted(
                (structural_tokens_batch, smiles_tokens_batch, sequence_id_batch),
                token_sampler=token_sampler,
            )
        all_sampled_tokens.append(tokens)
        all_token_probs.append(probs)
        # jxliu2: at the end of the batch, clear KV cache
        transformer_model.clear_kv_cache()
    # all_sampled_tokens = torch.cat(all_sampled_tokens, dim=0)
    # all_token_probs = torch.cat(all_token_probs, dim=0)
    return all_sampled_tokens, all_token_probs


def sample_components_from_transfusion_transformer(
    transformer_model: L.LightningModule,
    smiles_tokens: torch.Tensor,
    sequence_id: torch.Tensor,
    token_sampler: TokenSampler,
    diff_config: dict,
    alphabet: np.ndarray,
    inference_batch_size: int = 128,
    run_diffusion: bool = True,
) -> list[tuple[str, list[float], list[int], np.ndarray]]:
    """
    Samples from a trained transfusion model by first performing next-token prediction and
    then reverse diffusion sampling for the molecular dihedrals

    Args:
        transformer_model: L.LightnigModule
            The trained transfusion model loaded back into PyTorch Lightning
        smiles_tokens: torch.Tensor
            The starting tensor of SMILES tokens, containing the sequence start token
        sequence_id: torch.Tensor
            The sequence ID tensor which is binary (and all 1's to start with)
        diff_config: dict
            Dictionary containing additional options required for diffusion sampling
                with a trained transformer model
        alphabet: np.ndarray
            Array of unique tokens for decoding indices into SMILES strings
        inference_batch_size: int
            The batch size for inference. Default is 128
        run_diffusion: bool
            If torsional diffusion should be carried out to sample dihedral angles. Default is True

    Returns:
        predictions: list[tuple[str, list[float], list[int], np.ndarray]]
            A list of tuples containing the smiles strings, the dihedral angles computed from RDKit, the indices
                that the dihedrals map to in the molecule, and the torsions sampled by the transformer model
    """
    transformer_model.model.eval()
    assert not transformer_model.model.training
    all_final_predictions = []
    all_completed_structures = []
    num_batches = (
        smiles_tokens.shape[0] + inference_batch_size - 1
    ) // inference_batch_size
    for batch in range(num_batches):
        print(batch, num_batches)
        smiles_tokens_batch = smiles_tokens[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        sequence_id_batch = sequence_id[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        final_predictions, completed_structures = transformer_model.forward_transfusion(
            (smiles_tokens_batch, sequence_id_batch, alphabet, diff_config),
            token_sampler=token_sampler,
            run_diffusion=run_diffusion,
        )
        all_final_predictions.extend(final_predictions)
        all_completed_structures.extend(completed_structures)
    return all_final_predictions, all_completed_structures


def sample_structure_from_transfusion_transformer(
    transformer_model: L.LightningModule,
    smiles_tokens: list[torch.Tensor],
    diff_config: dict,
    alphabet: np.ndarray,
    inference_batch_size: int = 128,
) -> list[tuple[str, list[float], list[int], np.ndarray]]:
    """Samples dihedral angles for provided SMILES strings using a trained transfusion model

    Args:
        transformer_model: L.LightnigModule
            The trained transfusion model loaded back into PyTorch Lightning
        smiles_tokens: torch.Tensor
            The molecules to sample dihedral angles for, expressed as a list of SMILES tokens
        diff_config: dict
            Dictionary containing additional options required for diffusion sampling
                with a trained transformer model
        alphabet: np.ndarray
            Array of unique tokens for decoding indices into SMILES strings
        inference_batch_size: int
            The batch size for inference. Default is 128
        run_diffusion: bool
            If torsional diffusion should be carried out to sample dihedral angles. Default is True

    Returns:
        predictions: list[tuple[str, list[float], list[int], np.ndarray]]
            A list of tuples containing the smiles strings, the dihedral angles computed from RDKit, the indices
                that the dihedrals map to in the molecule, and the torsions sampled by the transformer model

    Notes:
        No sequence_id is required because the sequence id is constructed on the fly for diffusion
        smiles_tokens is a list because molecules can have different numbers of tokens (way to get around ragged tensor)
    """
    transformer_model.model.eval()
    assert not transformer_model.model.training
    all_final_predictions = []
    num_batches = (
        len(smiles_tokens) + inference_batch_size - 1
    ) // inference_batch_size
    for batch in range(num_batches):
        print(batch, num_batches)
        smiles_tokens_batch = smiles_tokens[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        final_predictions, _ = transformer_model.forward_transfusion_structure_only(
            (smiles_tokens_batch, alphabet, diff_config)
        )
        all_final_predictions.extend(final_predictions)
    return all_final_predictions, smiles_tokens


def sample_smiles_structure_from_mixed_seq_transformer(
    transformer_model: L.LightningModule,
    smiles_tokens: list[torch.Tensor],
    sequence_id: torch.Tensor,
    token_sampler: TokenSampler,
    inference_batch_size: int = 128,
    alphabet: np.ndarray = None,
    mode: str = "masked",
    struct_limit_method: str = "n_atoms",
    struct_limit_multiplier: float = 2.5,
    use_finished_smiles: bool = False,
    avh_args: dict = None,
):
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
    all_smiles, all_structures = [], []
    for batch in range(num_batches):
        if batch % 100 == 0:
            print(batch, num_batches)
        smiles_tokens_batch = smiles_tokens[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        sequence_id_batch = sequence_id[
            batch * inference_batch_size : (batch + 1) * inference_batch_size
        ]
        sampled_smiles_tokens, sampled_structure_tokens = (
            transformer_model.forward_seq_struct_mixed(
                (
                    smiles_tokens_batch.to(device)
                    if isinstance(smiles_tokens_batch, torch.Tensor)
                    else [x.to(device) for x in smiles_tokens_batch],
                    sequence_id_batch.to(device),
                ),
                token_sampler=token_sampler,
                alphabet=alphabet,
                mode=mode,
                struct_limit_method=struct_limit_method,
                struct_limit_multiplier=struct_limit_multiplier,
                use_finished_smiles=use_finished_smiles,
                avh_args=avh_args,
            )
        )
        all_smiles.append(sampled_smiles_tokens)
        all_structures.append(sampled_structure_tokens)
    return all_smiles, all_structures
