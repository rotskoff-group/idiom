"""Shared pytest config: force CPU and seed, so the suite runs on the shared box."""

import os

# Must be set before any idiom import resolves a device.
os.environ.setdefault("IDIOM_DEVICE", "cpu")

import pytest  # noqa: E402
import torch  # noqa: E402

import reward_fixtures  # noqa: E402,F401  registers fraction_proline/fraction_alanine for tests


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
