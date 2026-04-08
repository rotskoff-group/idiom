from protgps.utils.registry import register_object
import pytorch_lightning as pl
from protgps.utils.classes import ProtGPS


@register_object("wandb", "logger")
class WandB(pl.loggers.WandbLogger, ProtGPS):
    def __init__(self, args) -> None:
        super().__init__(
            project=args.project_name,
            name=args.experiment_name,
            entity=args.workspace,
            tags=args.logger_tags,
        )

    def setup(self, **kwargs):
        # "gradients", "parameters", "all", or None
        # # change "log_freq" log frequency of gradients and parameters (100 steps by default)
        if kwargs["args"].local_rank == 0:
            self.watch(kwargs["model"], log="all")
            self.experiment.config.update(kwargs["args"])

    def log_image(self, image, name):
        self.log_image(images=[image], caption=[name])
