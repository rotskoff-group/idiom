"""KV-cached generation with temperature, top-k, and nucleus sampling."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from idiom.data.tokenizer import Tokenizer
from idiom.model.attention import KVCache
from idiom.utils.validation import integer_at_least, validate_sampling


def _filter_top_k(logits: Tensor, k: int | None) -> Tensor:
    """Mask logits below the kth-largest value, retaining ties at the cutoff."""
    if not k or k >= logits.size(-1):
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def _filter_top_p(logits: Tensor, p: float | None) -> Tensor:
    """Keep the smallest descending-probability prefix whose cumulative mass reaches p."""
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    remove = cum > p
    remove[..., 1:] = remove[..., :-1].clone()  # shift so the token that crosses p is kept
    remove[..., 0] = False
    remove = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)  # back to original order
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token per logits row.

    Apply temperature, top-k, then top-p. Temperature 0 uses argmax without filtering.

    Args:
        logits: Logits of shape [B, V].
        temperature: Sampling temperature; 0 selects the argmax.
        top_k: If set, restrict sampling to the top_k highest-logit tokens.
        top_p: If set, restrict sampling to the smallest set of tokens whose cumulative probability
            exceeds top_p.
        generator: RNG for reproducible sampling.

    Returns:
        One sampled token id per row, shape [B].
    """
    validate_sampling(temperature, top_k, top_p)
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = _filter_top_p(_filter_top_k(logits / temperature, top_k), top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model,
    prompt_tokens: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_id: int | str | None = "auto",
    tokenizer: Tokenizer | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate continuations, prepending START internally.

    Finished rows retain stop_id and receive PAD thereafter. Stop once all rows finish.

    Args:
        model: The transformer to sample from.
        prompt_tokens: Prompt token ids of shape [B, P]; START is prepended internally.
        max_new_tokens: Maximum tokens to generate, capped by the remaining model context.
        temperature: Sampling temperature; 0 selects the argmax.
        top_k: Top-k filtering cutoff, or None.
        top_p: Top-p (nucleus) filtering cutoff, or None.
        stop_id: "auto" uses the tokenizer's STOP id, an int uses that id, and None disables early
            stopping so generation runs to the requested or context-limited token budget.
        tokenizer: Defaults to Tokenizer().
        generator: RNG for reproducible sampling.

    Returns:
        Generated token ids of shape [B, T], where T <= max_new_tokens.
    """
    integer_at_least("max_new_tokens", max_new_tokens, 1)
    validate_sampling(temperature, top_k, top_p)
    tok = tokenizer or Tokenizer()
    if stop_id == "auto":
        stop_id = tok.stop_id
    device = next(model.parameters()).device
    prompt_tokens = prompt_tokens.to(device)
    B = prompt_tokens.size(0)

    start = torch.full((B, 1), tok.start_id, dtype=torch.long, device=device)
    inp = torch.cat([start, prompt_tokens], dim=1)

    context_length = model.cfg.max_seq_len
    if inp.size(1) > context_length:
        raise ValueError(
            f"Prompt including START has {inp.size(1)} tokens, exceeding context length {context_length}"
        )
    # The last supported input position can predict one final token without another forward
    available = context_length - inp.size(1) + 1
    if max_new_tokens > available:
        warnings.warn(
            f"max_new_tokens={max_new_tokens} exceeds remaining context; limiting to {available} tokens",
            stacklevel=2,
        )
        max_new_tokens = available

    cache = KVCache(model.cfg.n_layers)
    logits = model(inp, cache=cache)[:, -1]

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated: list[Tensor] = []
    for step in range(max_new_tokens):
        nxt = sample_next_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        if stop_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, tok.pad_id), nxt)
            finished = finished | (nxt == stop_id)
        generated.append(nxt)
        if stop_id is not None and bool(finished.all()):
            break
        if step + 1 < max_new_tokens:
            logits = model(nxt[:, None], cache=cache)[:, -1]

    return torch.stack(generated, dim=1)
