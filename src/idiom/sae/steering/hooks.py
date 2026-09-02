"""Forward hooks that modify the residual stream, and the context manager that installs them.

Each transformer block returns its residual-stream output of shape [B, L, d_model]. A forward hook
registered on model.blocks[layer] returns a modified tensor, which becomes the input to the next
block. The hooks here cover four kinds of edit:

- add_direction_hook, add_relative_direction_hook, add_relative_renorm_direction_hook: add a
  vector, sized absolutely, relative to the local residual norm, or relative with the norm
  preserved;
- subtract_contribution_hook: remove chosen features' decoder contribution;
- sae_edit_hook: replace the residual with the SAE reconstruction of edited latents;
- substitute_hook: replace the residual with a fixed vector.

The steering context manager installs a hook and, given a tokenizer, confines its effect to the
positions selected by the tokenizer's region mask.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager

import torch


def add_direction_hook(direction: torch.Tensor, strength: float = 1.0) -> Callable:
    """Build a hook that adds strength * direction to every position's residual vector.

    Args:
        direction (torch.Tensor): The vector to add, shape [d_model].
        strength (float): Scalar multiplier on direction.

    Returns:
        Callable: A forward hook.
    """

    def hook(module, inputs, output):
        d = direction.to(output.device, output.dtype)
        return output + strength * d

    return hook


def add_relative_direction_hook(direction: torch.Tensor, alpha: float = 1.0) -> Callable:
    """Build a hook that adds alpha * ||x|| * unit(direction) at each position.

    The added vector is scaled by that position's own residual norm, so alpha is dimensionless.

    Args:
        direction (torch.Tensor): The steering direction, normalized internally, shape [d_model].
        alpha (float): Push size as a fraction of each position's residual norm.

    Returns:
        Callable: A forward hook.
    """
    u = direction / (direction.norm() + 1e-8)

    def hook(module, inputs, output):
        d = u.to(output.device, output.dtype)
        scale = alpha * output.norm(dim=-1, keepdim=True)   # [B, L, 1] per-position residual norm
        return output + scale * d

    return hook


def add_relative_renorm_direction_hook(direction: torch.Tensor, alpha: float = 1.0) -> Callable:
    """Build a hook that adds a relative direction, then restores each position's original norm.

    The output has the same per-position norm as the input, so the edit changes direction only.

    Args:
        direction (torch.Tensor): The steering direction, normalized internally, shape [d_model].
        alpha (float): How far to rotate toward the direction.

    Returns:
        Callable: A forward hook.
    """
    u = direction / (direction.norm() + 1e-8)

    def hook(module, inputs, output):
        d = u.to(output.device, output.dtype)
        norm = output.norm(dim=-1, keepdim=True)            # [B, L, 1] original per-position norm
        steered = output + alpha * norm * d                 # relative push (would grow the norm) ...
        return steered / (steered.norm(dim=-1, keepdim=True) + 1e-8) * norm  # ... then renorm back

    return hook


def subtract_contribution_hook(sae, feature_idxs: Sequence[int], scale: float = 1.0) -> Callable:
    """Build a hook computing x - scale * sum_f act_f(x) * W_dec[f] over the chosen features.

    Only the selected features' decoder contribution is removed; every other feature and the
    reconstruction residual are left untouched, and nothing is removed at positions where the
    feature did not fire. scale of 1 erases the features exactly, above 1 over-subtracts, and below
    0 amplifies them.

    Args:
        sae: The SAE providing encode_dense and W_dec.
        feature_idxs (Sequence[int]): Latents to subtract.
        scale (float): Multiplier on the subtracted contribution.

    Returns:
        Callable: A forward hook.
    """
    idx = torch.as_tensor(list(feature_idxs), dtype=torch.long)

    def hook(module, inputs, output):
        f = sae.encode_dense(output)            # [B, L, num_latents]
        j = idx.to(output.device)
        contrib = f[..., j] @ sae.W_dec[j]      # [B, L, d_model] = sum_k act_k * W_dec[k]
        return output - scale * contrib.to(output.dtype)

    return hook


def substitute_hook(vector: torch.Tensor) -> Callable:
    """Build a hook that replaces every position's residual vector with a fixed vector.

    Args:
        vector (torch.Tensor): The replacement vector, shape [d_model], broadcast over positions.

    Returns:
        Callable: A forward hook.
    """

    def hook(module, inputs, output):
        v = vector.to(output.device, output.dtype)
        return v.expand_as(output).clone()

    return hook


def sae_edit_hook(sae, edit_fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
    """Build a hook computing decode_dense(edit_fn(encode_dense(x))).

    The residual is replaced by the SAE reconstruction, so the output also carries the SAE's
    reconstruction error.

    Args:
        sae: The SAE providing encode_dense and decode_dense.
        edit_fn (Callable[[torch.Tensor], torch.Tensor]): Maps a dense latent tensor of shape
            [B, L, num_latents] to an edited tensor of the same shape.

    Returns:
        Callable: A forward hook.
    """

    def hook(module, inputs, output):
        f = sae.encode_dense(output)
        f = edit_fn(f)
        return sae.decode_dense(f).to(output.dtype)

    return hook


def clamp_features_edit(
    feature_idxs: Sequence[int], values: Sequence[float]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an edit function that sets each listed latent to a fixed value at every position.

    Args:
        feature_idxs (Sequence[int]): Latents to clamp.
        values (Sequence[float]): Target value for each latent, aligned with feature_idxs.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: An edit function over the dense latent tensor.

    Raises:
        ValueError: If feature_idxs and values differ in length.
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
    """Register a forward hook on one transformer block for the duration of the context.

    With a tokenizer given, the edit is confined to the positions the tokenizer's region mask
    selects, recomputed on every forward pass. A pre-hook accumulates the running token sequence
    across forwards, so the mask is correct during KV-cached decoding, when each forward sees only
    the newest token. With tokenizer None the hook applies at every position.

    Args:
        model: An IDiomTransformer.
        layer (int): Index of the block to hook.
        hook (Callable): The forward hook to register.
        tokenizer: Tokenizer used to recompute the region mask, or None to apply the hook
            everywhere.
        region (str): Positions to edit: "all", "idr", or "non_idr".

    Yields:
        The model, with the hook registered; all hooks are removed on exit.

    Example:
        with steering(model, layer=6, hook=add_direction_hook(sae.W_dec[j], 8.0)):
            logits = model(tokens)
    """
    handles = []
    if tokenizer is not None:
        latest: dict = {}

        def _capture(_module, args):
            tok_in = args[0]  # model(tokens, ...) -> args[0] is the token ids
            prev = latest.get("tokens")
            # A multi-token call (prefill / full forward) is the full context; a single new token
            # is an incremental cached decode step, so append it to the running sequence.
            if prev is None or tok_in.shape[1] > 1:
                latest["tokens"] = tok_in
            else:
                latest["tokens"] = torch.cat([prev, tok_in.to(prev.device)], dim=1)

        handles.append(model.register_forward_pre_hook(_capture))
        inner = hook

        def _masked(module, inputs, output):
            edited = inner(module, inputs, output)
            tokens = latest.get("tokens")
            n = output.shape[1]
            if tokens is None or tokens.shape[1] < n:
                return edited  # can't align tokens to positions; fall back to unmasked edit
            mask = tokenizer.region_mask(tokens, region=region)[:, -n:]  # last n positions
            return torch.where(mask.unsqueeze(-1).to(output.device), edited, output)

        hook = _masked

    handles.append(model.blocks[layer].register_forward_hook(hook))
    try:
        yield model
    finally:
        for h in reversed(handles):
            h.remove()
