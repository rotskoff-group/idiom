"""P0 SAE-port smoke tests (CPU-only): the merged SAE core works end to end."""

import torch

from idiom.data.tokens import residue_position_mask, residue_token_ids
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


def test_residue_mask():
    # ids 0,1,2 are FIM markers ('1','2','3'); 3,4,5 are real residues.
    token_info = {"alphabet": ["1", "2", "3", "A", "C", "D"]}
    assert residue_token_ids(token_info) == [3, 4, 5]

    toks = torch.tensor([[0, 3, 4, 1, 5, 2]])
    mask = residue_position_mask(toks, token_info)
    assert mask.tolist() == [[False, True, True, False, True, False]]
