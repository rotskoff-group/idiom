"""Shared pytest config: force CPU and seed, so the suite runs on the shared box."""

import os

# Must be set before any idiom import resolves a device.
os.environ.setdefault("IDIOM_DEVICE", "cpu")

import pytest
import torch

import reward_fixtures  # registers fraction_proline/fraction_alanine for tests


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
