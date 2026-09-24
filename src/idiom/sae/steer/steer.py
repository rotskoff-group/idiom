"""Generate IDRs by adding decoder directions, clamping latents, or ablating features."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.model.sampling import generate
from idiom.sae.steer.hooks import (
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
    """Feature-steering settings for a zero-based transformer block index.

    Attributes:
        layer: Block to edit.
        feature_idx: One feature index or a sequence of indices.
        strength: Scalar or per-feature weights for plain addition and clamping. Normalized/relative
            addition uses only the first value. Ablation requires a scalar subtraction factor:
            0 leaves activations unchanged, 1 removes the selected features' contributions.
        mode: "add_direction", "clamp" (SAE reconstruction), or "ablate" (subtract contribution).
        clamp_value: Clamp targets; defaults to strength. Scalars broadcast across features.
        normalize: Normalize the summed decoder direction before scaling by strength.
        relative: Scale by each residual norm; overrides normalize.
        preserve_norm: Restore original residual norms after relative addition only.
    """

    layer: int
    feature_idx: int | Sequence[int]
    strength: float | Sequence[float]
    mode: str = "add_direction"
    clamp_value: float | Sequence[float] | None = None
    normalize: bool = False
    relative: bool = False
    preserve_norm: bool = False


def _as_list(x) -> list:
    """Return x as a plain list, treating a str or a non-iterable as a single element."""
    if isinstance(x, str) or not hasattr(x, "__iter__"):
        return [x]
    return list(x)


def _broadcast(values: list, n: int, name: str) -> list:
    """Broadcast one value to n entries; reject lengths other than 1 or n."""
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name} has {len(values)} values but {n} feature(s) were given.")
    return values


def build_steering_hook(sae, spec: SteeringSpec) -> Callable:
    """Return the forward hook defined by spec; see SteeringSpec for mode semantics.

    Args:
        sae: The trained SAE providing W_dec, encode_dense, and decode_dense.
        spec: Which features to steer, how hard, and in which mode.

    Returns:
        The forward hook implementing the requested steering.

    Raises:
        ValueError: If spec.mode is not "add_direction", "clamp", or "ablate", or if a per-feature
            list length does not match the number of features.
    """
    feats = _as_list(spec.feature_idx)
    if spec.mode == "add_direction":
        if spec.relative:
            raw = sum(sae.W_dec[i].detach() for i in feats)
            alpha = float(_as_list(spec.strength)[0])
            if spec.preserve_norm:
                return add_relative_renorm_direction_hook(raw, alpha)
            return add_relative_direction_hook(raw, alpha)
        if spec.normalize:
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
        scale = float(spec.strength)
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
) -> torch.Tensor:
    """Sample with region-masked steering on spec.layer, updating the mask at each step.

    Args:
        model: An IDiomTransformer.
        sae: The trained SAE for spec.layer, providing W_dec, encode_dense, and decode_dense.
        spec: Which features to steer, how hard, and in which mode.
        prompt_tokens: A 1-D sequence of prompt token ids, repeated to n_samples; the encoding of
            "132" if None. START is prepended by the sampler.
        n_samples: Number of sequences to generate.
        max_new_tokens: Maximum new tokens to sample per sequence.
        temperature: Sampling temperature.
        top_k: Top-k sampling cutoff, or None.
        top_p: Nucleus sampling cutoff, or None.
        region: Positions to steer: "all", "idr", or "non_idr".
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
        prompt_tokens = tok.encode("132")
    prompt = torch.as_tensor(list(prompt_tokens), dtype=torch.long)
    prompts = prompt.unsqueeze(0).repeat(n_samples, 1).to(device)

    hook = build_steering_hook(sae, spec)
    with steering(model, spec.layer, hook, tokenizer=tok, region=region):
        return generate(
            model,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tokenizer=tok,
            generator=generator,
        )
