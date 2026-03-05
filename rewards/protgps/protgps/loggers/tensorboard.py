from protgps.utils.registry import register_object
import pytorch_lightning as pl
from protgps.utils.classes import ProtGPS


@register_object("tensorboard", "logger")
class PLTensorBoardLogger(pl.loggers.TensorBoardLogger, ProtGPS):
    def __init__(self, args) -> None:
        super().__init__(args.logger_dir)

    def setup(self, **kwargs):
        pass

    def log_image(self, image, name):
        pass

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--logger_dir",
            type=str,
            default=".",
            help="directory to save tensorboard logs",
        )
