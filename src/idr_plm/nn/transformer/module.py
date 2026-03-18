import lightning as L
import torch
import torch.nn as nn
import pl_bolts
from lightning.pytorch.utilities import grad_norm

from idr_plm.nn.transformer.nn import GeometricMolTransformer
from idr_plm.utils.sampler import TokenSampler
from idr_plm.nn.transformer.losses.autoreg_loss import shared_eval_autoreg
from idr_plm.nn.transformer.losses.grpo_loss import shared_eval_grpo


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
                    reduction="none",
                    ignore_index=self.token_info["input"]["TOK"]["TOK_PAD"],
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
        """Autoregressive loss evaluation, from autoreg_loss module"""
        return shared_eval_autoreg(self, batch, batch_idx, prefix)

    def _shared_eval_grpo(self, batch, batch_idx, prefix):
        """GRPO loss evaluation, from grpo_loss module"""
        return shared_eval_grpo(self, batch, batch_idx, prefix)

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
