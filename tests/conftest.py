"""Shared pytest config: force CPU and seed, so the suite runs on the shared box."""

import os

# Must be set before any idiom import resolves a device
os.environ.setdefault("IDIOM_DEVICE", "cpu")

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    """Reset the PyTorch random seed before each test."""
    torch.manual_seed(0)
