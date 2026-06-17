"""Forward-hook injection into IDiom's residual stream.

Each IDiom transformer block returns its residual-stream output ``x`` of shape
``[B, L, d_model]``. A PyTorch forward hook on ``model.blocks[layer]`` can return a modified
tensor, which becomes the input to the next block — a clean way to patch / steer the residual
stream without touching idiom's source (no nnsight needed, unlike InterPLM's ESM path).

Two steering primitives:
- :func:`add_direction_hook`   add a fixed vector (e.g. a scaled SAE decoder column).
- :func:`sae_edit_hook`        encode with the SAE, edit latents, decode, and substitute.

SAEs are trained only on a region of the residual stream (residues, or just the IDR / flanks;
START / FIM-marker / control positions are always dropped), so edits are confined to that same
region. Pass a ``tokenizer`` (and optional ``region``) to :func:`steering` and it recomputes the
:meth:`~idiom.data.tokenizer.Tokenizer.region_mask` on every forward, which is what
autoregressive generation needs as the sequence grows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager

import torch


def add_direction_hook(direction: torch.Tensor, strength: float = 1.0) -> Callable:
    """Hook that adds ``strength * direction`` to every position's residual vector."""

    def hook(module, inputs, output):
        d = direction.to(output.device, output.dtype)
        return output + strength * d

    return hook


def substitute_hook(vector: torch.Tensor) -> Callable:
    """Hook that replaces the entire residual stream with a fixed vector (broadcast).

    Used as the mean-ablation baseline in fidelity eval: every position's residual becomes
    ``vector`` (e.g. the dataset mean or ``sae.b_dec``), destroying per-token information.
    """

    def hook(module, inputs, output):
        v = vector.to(output.device, output.dtype)
        return v.expand_as(output).clone()

    return hook


def sae_edit_hook(sae, edit_fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
    """Hook that replaces the residual stream with the SAE reconstruction after editing latents.

    ``edit_fn`` maps the dense latent tensor ``[B, L, num_latents]`` to an edited latent
    tensor of the same shape (e.g. clamp feature j to a target value, or zero it out). The
    residual is then ``decode_dense(edit_fn(encode_dense(x)))``. Note this also incurs the
    SAE reconstruction error.
    """

    def hook(module, inputs, output):
        f = sae.encode_dense(output)
        f = edit_fn(f)
        return sae.decode_dense(f).to(output.dtype)

    return hook


def clamp_features_edit(
    feature_idxs: Sequence[int], values: Sequence[float]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an ``edit_fn`` that clamps several latents at once (one SAE round-trip).

    ``feature_idxs[k]`` is set to ``values[k]`` at every position. The edit acts on the full
    dense latent tensor, so clamping N features costs the same encode/decode as clamping one.
    """
    if len(feature_idxs) != len(values):
        raise ValueError(f"feature_idxs ({len(feature_idxs)}) and values ({len(values)}) differ in length.")
    idx = torch.as_tensor(list(feature_idxs), dtype=torch.long)
    val = torch.as_tensor(list(values), dtype=torch.float)

    def edit(f: torch.Tensor) -> torch.Tensor:
        f = f.clone()
        f[..., idx.to(f.device)] = val.to(f.device, f.dtype)
        return f

    return edit


@contextmanager
def steering(model, layer: int, hook: Callable, *, tokenizer=None, region: str = "all"):
    """Register ``hook`` on ``model.blocks[layer]`` (an :class:`IDiomTransformer`) for the block.

    When ``tokenizer`` is given, the edit is confined to the SAE's training region: each forward
    the mask is recomputed from the model's input tokens via
    :meth:`~idiom.data.tokenizer.Tokenizer.region_mask` (``region``), so START / FIM-marker /
    control positions — and, for ``idr``/``non_idr``, the wrong side of the ``2`` marker — pass
    through unmodified. This keeps the SAE on the distribution it was trained on and tracks the
    growing token sequence during autoregressive generation. With ``tokenizer=None`` the hook
    applies at every position.

    Example::

        with steering(model, layer=6, hook=add_direction_hook(sae.W_dec[j], 8.0)):
            logits = model(tokens)
    """
    handles = []
    if tokenizer is not None:
        latest: dict = {}

        def _capture(_module, args):
            latest["tokens"] = args[0]  # model(tokens, ...) -> args[0] is the token ids

        handles.append(model.register_forward_pre_hook(_capture))
        inner = hook

        def _masked(module, inputs, output):
            edited = inner(module, inputs, output)
            tokens = latest.get("tokens")
            if tokens is None or tokens.shape[1] != output.shape[1]:
                return edited  # can't align tokens to positions; fall back to unmasked edit
            mask = tokenizer.region_mask(tokens, region=region).unsqueeze(-1).to(output.device)
            return torch.where(mask, edited, output)

        hook = _masked

    handles.append(model.blocks[layer].register_forward_hook(hook))
    try:
        yield model
    finally:
        for h in reversed(handles):
            h.remove()
