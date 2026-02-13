from typing import Sequence, TypeVar
import random
import numpy as np
import torch

MAX_SUPPORTED_DISTANCE = 1e6

TSequence = TypeVar("TSequence", bound=Sequence)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
