"""Residual-stream edit hooks and a context manager for region-masked steering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager

import torch


def add_direction_hook(direction: torch.Tensor, strength: float = 1.0) -> Callable:
    """Return a hook adding strength * direction ([d_model]) at every position."""

    def hook(module, inputs, output):
        """Add the scaled steering direction to the layer output."""
        d = direction.to(output.device, output.dtype)
        return output + strength * d

    return hook


def add_relative_direction_hook(direction: torch.Tensor, alpha: float = 1.0) -> Callable:
    """Return a hook adding alpha * ||x|| * unit(direction) at each position.

    Direction has shape [d_model]; alpha is a fraction of the local residual norm.
    """
    u = direction / (direction.norm() + 1e-8)

    def hook(module, inputs, output):
        """Add the steering direction scaled by each output vector norm."""
        d = u.to(output.device, output.dtype)
        scale = alpha * output.norm(dim=-1, keepdim=True)
        return output + scale * d

    return hook


def add_relative_renorm_direction_hook(direction: torch.Tensor, alpha: float = 1.0) -> Callable:
    """Return a relative-direction hook that restores each position's original norm.

    Add alpha * ||x|| * unit(direction), then renormalize. Direction has shape [d_model].
    """
    u = direction / (direction.norm() + 1e-8)

    def hook(module, inputs, output):
        """Steer each output vector and restore its original norm."""
        d = u.to(output.device, output.dtype)
        norm = output.norm(dim=-1, keepdim=True)
        steered = output + alpha * norm * d
        return steered / (steered.norm(dim=-1, keepdim=True) + 1e-8) * norm

    return hook


def subtract_contribution_hook(sae, feature_idxs: Sequence[int], scale: float = 1.0) -> Callable:
    """Return a hook subtracting scale * sum_f activation_f(x) * W_dec[f].

    Use feature_idxs to select latents. Scale 1 removes their current decoder contribution;
    negative scale amplifies it. Preserve the reconstruction residual.
    """
    idx = torch.as_tensor(list(feature_idxs), dtype=torch.long)

    def hook(module, inputs, output):
        """Subtract the scaled decoder contributions of selected SAE features."""
        if scale == 0:
            return output
        f = sae.encode_dense(output)  # [B, L, num_latents]
        j = idx.to(output.device)
        contrib = f[..., j] @ sae.W_dec[j]  # [B, L, d_model] = sum_k act_k * W_dec[k]
        return output - scale * contrib.to(output.dtype)

    return hook


def substitute_hook(vector: torch.Tensor) -> Callable:
    """Return a hook replacing every residual vector with vector ([d_model])."""

    def hook(module, inputs, output):
        """Replace each output vector with a copy of the supplied vector."""
        v = vector.to(output.device, output.dtype)
        return v.expand_as(output).clone()

    return hook


def sae_edit_hook(sae, edit_fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
    """Return a hook computing sae.decode_dense(edit_fn(sae.encode_dense(x))).

    edit_fn maps [B, L, num_latents] to the same shape. Replacing the residual with the
    SAE reconstruction introduces reconstruction error.
    """

    def hook(module, inputs, output):
        """Encode the output, edit its dense SAE features, and decode the edited representation."""
        f = sae.encode_dense(output)
        f = edit_fn(f)
        return sae.decode_dense(f).to(output.dtype)

    return hook


def clamp_features_edit(
    feature_idxs: Sequence[int], values: Sequence[float]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build an edit function that sets each listed latent to a fixed value at every position.

    Args:
        feature_idxs: Latents to clamp.
        values: Target value for each latent, aligned with feature_idxs.

    Returns:
        An edit function over the dense latent tensor.

    Raises:
        ValueError: If feature_idxs and values differ in length.
    """
    if len(feature_idxs) != len(values):
        raise ValueError(f"feature_idxs ({len(feature_idxs)}) and values ({len(values)}) differ in length.")
    idx = torch.as_tensor(list(feature_idxs), dtype=torch.long)
    val = torch.as_tensor(list(values), dtype=torch.float)

    def edit(f: torch.Tensor) -> torch.Tensor:
        """Copy feature activations and set selected coordinates to the requested values."""
        f = f.clone()
        f[..., idx.to(f.device)] = val.to(f.device, f.dtype)
        return f

    return edit


@contextmanager
def steering(model, layer: int, hook: Callable, *, tokenizer=None, region: str = "all"):
    """Temporarily install a hook on model.blocks[layer]; remove hooks on exit.

    With a tokenizer, mask edits by region, tracking tokens during cached decoding.
    Without one, edit all positions.

    Args:
        model: An IDiomTransformer.
        layer: Index of the block to hook.
        hook: The forward hook to register.
        tokenizer: Tokenizer used to recompute the region mask, or None to apply the hook
            everywhere.
        region: Positions to edit: "all", "idr", or "non_idr".

    Yields:
        The model, with the hook registered; all hooks are removed on exit.

    Examples:
        with steering(model, layer=6, hook=add_direction_hook(sae.W_dec[j], 8.0)):
            logits = model(tokens)
    """
    handles = []
    if tokenizer is not None:
        latest: dict = {}

        def _capture(_module, args):
            """Track prompt and cached decoding tokens for residue-region masking."""
            tok_in = args[0]
            prev = latest.get("tokens")
            # Prefill replaces the context; cached single-token decoding extends it
            if prev is None or tok_in.shape[1] > 1:
                latest["tokens"] = tok_in
            else:
                latest["tokens"] = torch.cat([prev, tok_in.to(prev.device)], dim=1)

        handles.append(model.register_forward_pre_hook(_capture))
        inner = hook

        def _masked(module, inputs, output):
            """Apply the inner steering hook only at positions selected by the token-region mask."""
            edited = inner(module, inputs, output)
            tokens = latest.get("tokens")
            n = output.shape[1]
            if tokens is None or tokens.shape[1] < n:
                return edited  # can't align tokens to positions; fall back to unmasked edit
            mask = tokenizer.region_mask(tokens, region=region)[:, -n:]
            return torch.where(mask.unsqueeze(-1).to(output.device), edited, output)

        hook = _masked

    handles.append(model.blocks[layer].register_forward_hook(hook))
    try:
        yield model
    finally:
        for h in reversed(handles):
            h.remove()
