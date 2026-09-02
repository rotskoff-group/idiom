"""Feature-steered generation.

Runs the KV-cached sampler inside a steering context, so the chosen layer's residual stream is
modified at every forward pass of generation.

SteeringSpec selects the mode:

- "add_direction": add the selected features' decoder rows, scaled absolutely, relative to the
  local residual norm, or relative with the norm preserved;
- "clamp": encode, set the selected latents to fixed values, decode, and substitute;
- "ablate": subtract the selected features' decoder contribution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.model.sampling import generate
from idiom.sae.steering.hooks import (
    add_direction_hook,
    add_relative_direction_hook,
    add_relative_renorm_direction_hook,
    clamp_features_edit,
    sae_edit_hook,
    steering,
    subtract_contribution_hook,
)


@dataclass
class SteeringSpec:
    """Which features to steer, how hard, and in which mode.

    Attributes:
        layer (int): Residual-stream layer to steer.
        feature_idx (int | Sequence[int]): One latent index, or several to steer together.
        strength (float | Sequence[float]): Steering strength; a scalar applied to every feature,
            or one value per feature. Its meaning depends on mode and the flags below.
        mode (str): "add_direction", "clamp", or "ablate".
        clamp_value (float | Sequence[float] | None): Target activation for "clamp"; strength is
            used when None.
        normalize (bool): For "add_direction", scale the summed decoder rows to unit norm before
            applying strength, so strength sets the push magnitude directly.
        relative (bool): For "add_direction", scale the push by each position's residual norm, so
            strength is a fraction of it. Takes precedence over normalize.
        preserve_norm (bool): With relative, restore each position's original residual norm after
            the push.
    """

    layer: int
    feature_idx: int | Sequence[int]
    strength: float | Sequence[float]
    mode: str = "add_direction"  # "add_direction" | "clamp" | "ablate"
    clamp_value: float | Sequence[float] | None = None
    normalize: bool = False  # add_direction: use strength * unit(sum of decoder rows), so the push
    #                          magnitude == strength regardless of how many features are summed.
    relative: bool = False   # add_direction: push = strength * ||x_pos|| * unit(sum of decoder rows),
    #                          i.e. strength is a dimensionless FRACTION of the local residual norm.
    preserve_norm: bool = False  # relative only: renorm each position back to ||x_pos|| after the push,
    #                              so steering ROTATES x toward the feature at constant norm (no inflation).


def _as_list(x) -> list:
    """Return x as a plain list, treating a str or a non-iterable as a single element."""
    if isinstance(x, str) or not hasattr(x, "__iter__"):
        return [x]
    return list(x)


def _broadcast(values: list, n: int, name: str) -> list:
    """Return values repeated to length n if it holds one element, or unchanged if it holds n.

    Raises:
        ValueError: If values holds neither 1 nor n elements.
    """
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name} has {len(values)} values but {n} feature(s) were given.")
    return values


def build_steering_hook(sae, spec: SteeringSpec) -> Callable:
    """Build the forward hook described by a SteeringSpec.

    "add_direction" adds the sum of the selected decoder rows, scaled according to the spec's
    normalize, relative, and preserve_norm flags. "clamp" pins the selected latents to their target
    values in one SAE round trip. "ablate" subtracts the selected features' decoder contribution,
    using strength as the subtraction factor.

    Args:
        sae: The trained SAE providing W_dec, encode_dense, and decode_dense.
        spec (SteeringSpec): Which features to steer, how hard, and in which mode.

    Returns:
        Callable: The forward hook implementing the requested steering.

    Raises:
        ValueError: If spec.mode is not "add_direction", "clamp", or "ablate", or if a per-feature
            list length does not match the number of features.
    """
    feats = _as_list(spec.feature_idx)
    if spec.mode == "add_direction":
        if spec.relative:
            # push = strength(=alpha) * ||x_pos|| * unit(sum of unit decoder rows)
            raw = sum(sae.W_dec[i].detach() for i in feats)
            alpha = float(_as_list(spec.strength)[0])
            if spec.preserve_norm:  # rotate toward the feature at constant ||x_pos|| (no norm inflation)
                return add_relative_renorm_direction_hook(raw, alpha)
            return add_relative_direction_hook(raw, alpha)
        if spec.normalize:
            # strength sets the push magnitude directly: strength * unit(sum of unit decoder rows),
            # so N (how many features) controls only the direction, not the magnitude.
            raw = sum(sae.W_dec[i].detach() for i in feats)
            scale = float(_as_list(spec.strength)[0])
            direction = raw / (raw.norm() + 1e-8) * scale
        else:
            strengths = _broadcast(_as_list(spec.strength), len(feats), "strength")
            direction = sum(s * sae.W_dec[i].detach() for i, s in zip(feats, strengths))
        return add_direction_hook(direction, strength=1.0)
    if spec.mode == "clamp":
        raw = spec.clamp_value if spec.clamp_value is not None else spec.strength
        values = _broadcast(_as_list(raw), len(feats), "clamp_value")
        return sae_edit_hook(sae, clamp_features_edit(feats, values))
    if spec.mode == "ablate":
        scale = float(spec.strength) if spec.strength else 1.0  # strength == over-ablation factor (default 1)
        return subtract_contribution_hook(sae, feats, scale=scale)
    raise ValueError(f"Unknown steering mode {spec.mode!r}; use 'add_direction', 'clamp', or 'ablate'.")


@torch.no_grad()
def steer_generation(
    model,
    sae,
    spec: SteeringSpec,
    *,
    prompt_tokens=None,
    n_samples: int = 100,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    region: str = "all",
    tokenizer=None,
    generator=None,
):
    """Generate sequences with one or more SAE features steered.

    The sampler runs inside a steering context on spec.layer, so the edit is applied at every
    generation step and the region mask is recomputed as the sequence grows.

    Args:
        model: An IDiomTransformer.
        sae: The trained SAE for spec.layer, providing W_dec, encode_dense, and decode_dense.
        spec (SteeringSpec): Which features to steer, how hard, and in which mode.
        prompt_tokens: A 1-D sequence of prompt token ids, repeated to n_samples; the encoding of
            "132" if None. START is prepended by the sampler.
        n_samples (int): Number of sequences to generate.
        max_new_tokens (int): Maximum new tokens to sample per sequence.
        temperature (float): Sampling temperature.
        top_k (int | None): Top-k sampling cutoff, or None.
        top_p (float | None): Nucleus sampling cutoff, or None.
        region (str): Positions to steer: "all", "idr", or "non_idr".
        tokenizer: Tokenizer for encoding the prompt and building the region mask; a default if
            None.
        generator: torch.Generator for reproducible sampling, or None.

    Returns:
        Tensor: Generated token ids of shape [n_samples, T].
    """
    tok = tokenizer or Tokenizer()
    device = next(model.parameters()).device
    sae = sae.to(device).eval()

    if prompt_tokens is None:
        prompt_tokens = tok.encode("132")  # unprompted / de novo
    prompt = torch.as_tensor(list(prompt_tokens), dtype=torch.long)
    prompts = prompt.unsqueeze(0).repeat(n_samples, 1).to(device)

    hook = build_steering_hook(sae, spec)
    with steering(model, spec.layer, hook, tokenizer=tok, region=region):
        return generate(
            model, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, tokenizer=tok, generator=generator,
        )
