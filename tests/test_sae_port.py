"""P0 SAE core smoke test (CPU-only): encode/decode round-trip and top-k sparsity."""

import torch

from idiom.sae import SparseCoder


def test_sparse_coder_forward_cpu():
    d_in, k, n_latents = 16, 4, 64
    sae = SparseCoder(d_in, num_latents=n_latents, k=k)
    x = torch.randn(32, d_in)

    out = sae(x)
    assert out.sae_out.shape == x.shape
    assert torch.isfinite(out.fvu)

    f = sae.encode_dense(x)
    assert f.shape == (32, n_latents)
    assert sae.decode_dense(f).shape == x.shape
    # top-k sparsity: at most k strictly-positive latents per row.
    assert int((f > 0).sum(-1).max()) <= k
