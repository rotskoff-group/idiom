"""Residual-stream forward hooks and feature-steered generation."""

from idiom.sae.steer.hooks import (
    add_direction_hook,
    sae_edit_hook,
    steering,
    substitute_hook,
)
from idiom.sae.steer.steer import SteeringSpec, steer_generation

__all__ = [
    "SteeringSpec",
    "add_direction_hook",
    "sae_edit_hook",
    "steer_generation",
    "steering",
    "substitute_hook",
]
