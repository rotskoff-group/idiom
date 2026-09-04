"""KV-cached autoregressive generation.

START is prepended to the prompt, the prompt is prefilled in one forward pass that fills the
cache, and tokens are then decoded one at a time. Each sequence in a batch stops at its first STOP
token and is padded thereafter. Sampling is controlled by temperature (0 for greedy), top_k, and
top_p.
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
    """Sample one token id per row of logits.

    Filtering is applied after temperature scaling, top_k before top_p.

    Args:
        logits (Tensor): Logits of shape [B, V].
        temperature (float): Sampling temperature; 0 selects the argmax.
        top_k (int | None): If set, restrict sampling to the top_k highest-logit tokens.
        top_p (float | None): If set, restrict sampling to the smallest set of tokens whose
            cumulative probability exceeds top_p.
        generator (torch.Generator | None): RNG for reproducible sampling.

    Returns:
        Tensor: One sampled token id per row, shape [B].
    """
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
    """Generate tokens continuing each prompt in a batch.

    Decoding stops early once every sequence has emitted stop_id; sequences that finish first are
    filled with the tokenizer's PAD id.

    Args:
        model: The transformer to sample from.
        prompt_tokens (Tensor): Prompt token ids of shape [B, P]; START is prepended internally.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature; 0 selects the argmax.
        top_k (int | None): Top-k filtering cutoff, or None.
        top_p (float | None): Top-p (nucleus) filtering cutoff, or None.
        stop_id (int | str | None): "auto" uses the tokenizer's STOP id, an int uses that id, and
            None disables early stopping so exactly max_new_tokens are generated.
        tokenizer (Tokenizer | None): Tokenizer; a default Tokenizer if None.
        generator (torch.Generator | None): RNG for reproducible sampling.

    Returns:
        Tensor: Generated token ids of shape [B, T], where T <= max_new_tokens.
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
