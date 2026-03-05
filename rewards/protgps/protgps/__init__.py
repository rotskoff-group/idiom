# type: ignore

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


# data
import protgps.datasets.protein_compartments
import protgps.datasets.reverse_homology

# lightning
import protgps.lightning.base

# optimizers
import protgps.learning.optimizers.basic

# scheduler
import protgps.learning.schedulers.basic

# losses
import protgps.learning.losses.basic

# metrics
import protgps.learning.metrics.basic

# callbacks
import protgps.callbacks.basic
import protgps.callbacks.swa

# models
import protgps.models.classifier
import protgps.models.fair_esm

# comet
import protgps.loggers.comet
import protgps.loggers.wandb
import protgps.loggers.tensorboard
