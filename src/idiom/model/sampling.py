"""KV-cached autoregressive generation.

START is prepended to the prompt, the prompt is prefilled in one forward (populating the
cache), then tokens are decoded one at a time. Per-sequence early stop on the STOP token;
finished sequences are padded so a batch can finish at different lengths. Standard sampling
controls: ``temperature`` (0 = greedy), ``top_k``, ``top_p``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from idiom.data.tokenizer import Tokenizer
from idiom.model.attention import KVCache


def _filter_top_k(logits: Tensor, k: int | None) -> Tensor:
    if not k or k >= logits.size(-1):
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1, None]  # k-th largest logit per row
    return logits.masked_fill(logits < kth, float("-inf"))


def _filter_top_p(logits: Tensor, p: float | None) -> Tensor:
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    remove = cum > p  # drop the tail past cumulative prob p
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
    """Sample one token id per row from ``logits`` ``[B, V]``. ``temperature=0`` -> greedy."""
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
    """Generate up to ``max_new_tokens`` tokens after ``prompt_tokens`` ``[B, P]``.

    Returns the generated ids ``[B, T]`` (T ≤ max_new_tokens), padded after STOP. ``stop_id``:
    ``"auto"`` uses the tokenizer's STOP, ``None`` disables early stopping (fixed length).
    """
    tok = tokenizer or Tokenizer()
    if stop_id == "auto":
        stop_id = tok.stop_id
    device = next(model.parameters()).device
    prompt_tokens = prompt_tokens.to(device)
    B = prompt_tokens.size(0)

    start = torch.full((B, 1), tok.start_id, dtype=torch.long, device=device)
    inp = torch.cat([start, prompt_tokens], dim=1)  # prepend START

    cache = KVCache(model.cfg.n_layers)
    logits = model(inp, cache=cache)[:, -1]  # prefill -> last-position logits

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated: list[Tensor] = []
    for _ in range(max_new_tokens):
        nxt = sample_next_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        if stop_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, tok.pad_id), nxt)  # pad once finished
            finished = finished | (nxt == stop_id)
        generated.append(nxt)
        if stop_id is not None and bool(finished.all()):
            break
        logits = model(nxt[:, None], cache=cache)[:, -1]  # decode one step via the cache

    return torch.stack(generated, dim=1)
