"""Feature-steered IDiom generation.

Goal: bias IDiom's IDR generation toward (or away from) the biology a chosen SAE feature
encodes — e.g. push a base-model generation toward a protGPS condensate by amplifying the
feature that the corresponding RL model upregulates.

This wires :mod:`idiom.sae.steering.hooks` into idiom's autoregressive sampler. The
sampling loop is the KV-cached :func:`idiom.model.sampling.generate`; we run it inside a
:func:`idiom.sae.steering.hooks.steering` context so the chosen layer's residual stream is
steered on every forward pass of generation.

Two steering modes (see :class:`SteeringSpec`):
- ``"add_direction"``  add ``strength * sae.W_dec[feature_idx]`` to every position. Cheap,
  surgical, leaves all other features untouched. The decoder rows are unit-norm, so
  ``strength`` is in residual-norm units.
- ``"clamp"``          encode with the SAE, set latent ``feature_idx`` to ``clamp_value``
  everywhere, decode, and substitute. Heavier-handed (also incurs SAE reconstruction error)
  but pins the feature to an exact activation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from idiom.data.tokenizer import Tokenizer
from idiom.sae.steering.hooks import (
    add_direction_hook,
    clamp_features_edit,
    sae_edit_hook,
    steering,
)


@dataclass
class SteeringSpec:
    """Which feature(s) to steer, how hard, and in which mode.

    ``feature_idx`` may be a single index or a list of indices to steer simultaneously.
    ``strength`` / ``clamp_value`` are either a scalar (applied to every feature) or a list
    aligned with ``feature_idx`` (one value per feature).
    """

    layer: int
    feature_idx: int | Sequence[int]
    strength: float | Sequence[float]
    mode: str = "add_direction"  # "add_direction" | "clamp"
    clamp_value: float | Sequence[float] | None = None


def _as_list(x) -> list:
    """Normalise a scalar / list / omegaconf ListConfig to a plain list (str stays scalar)."""
    if isinstance(x, str) or not hasattr(x, "__iter__"):
        return [x]
    return list(x)


def _broadcast(values: list, n: int, name: str) -> list:
    """Broadcast a length-1 list to ``n``, or pass through if already length ``n``."""
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name} has {len(values)} values but {n} feature(s) were given.")
    return values


def build_steering_hook(sae, spec: SteeringSpec) -> Callable:
    """Build the forward hook for ``spec`` from the trained ``sae``.

    Supports one or several features at once. ``add_direction`` adds the sum of the (unit-norm)
    decoder rows scaled by their strengths; ``clamp`` rewrites the residual via a single SAE
    encode/edit/decode round-trip that pins all listed features to their target values.
    """
    feats = _as_list(spec.feature_idx)
    if spec.mode == "add_direction":
        strengths = _broadcast(_as_list(spec.strength), len(feats), "strength")
        direction = sum(s * sae.W_dec[i].detach() for i, s in zip(feats, strengths))
        return add_direction_hook(direction, strength=1.0)
    if spec.mode == "clamp":
        raw = spec.clamp_value if spec.clamp_value is not None else spec.strength
        values = _broadcast(_as_list(raw), len(feats), "clamp_value")
        return sae_edit_hook(sae, clamp_features_edit(feats, values))
    raise ValueError(f"Unknown steering mode {spec.mode!r}; use 'add_direction' or 'clamp'.")


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
    """Generate IDRs from an :class:`IDiomTransformer` with SAE feature(s) steered.

    Runs the KV-cached sampler inside a steering context on ``spec.layer`` so the chosen
    feature is steered at every generation step.

    Args:
        model: an :class:`IDiomTransformer`.
        sae: trained SAE for ``spec.layer`` (provides ``W_dec`` / ``encode_dense`` / ``decode_dense``).
        spec: which feature(s) to steer, how hard, and in which mode.
        prompt_tokens: FIM prompt ids (1-D); defaults to ``encode("132")`` (de-novo IDP).
            Repeated to ``n_samples``. START is prepended by the sampler.
        n_samples: number of sequences to generate.
        max_new_tokens / temperature / top_k / top_p: sampler settings.
        region: the SAE's training region (``"all"`` | ``"idr"`` | ``"non_idr"``). The edit is
            always confined to it — the SAE was trained solely on those residue activations, so
            steering markers/START (or the wrong side of the ``2``) would apply it
            off-distribution. Recomputed each step as the sequence grows.

    Returns:
        Generated token ids ``[n_samples, T]`` (decode with the tokenizer).
    """
    from idiom.model.sampling import generate  # noqa: PLC0415

    tok = tokenizer or Tokenizer()
    device = next(model.parameters()).device
    sae = sae.to(device).eval()

    if prompt_tokens is None:
        prompt_tokens = tok.encode("132")  # de-novo IDP
    prompt = torch.as_tensor(list(prompt_tokens), dtype=torch.long)
    prompts = prompt.unsqueeze(0).repeat(n_samples, 1).to(device)

    hook = build_steering_hook(sae, spec)
    with steering(model, spec.layer, hook, tokenizer=tok, region=region):
        return generate(
            model, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, tokenizer=tok, generator=generator,
        )
