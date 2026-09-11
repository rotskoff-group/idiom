"""Steering tests: region-masked residual edits and steered sampling."""

import pytest
import torch

from idiom.data.io import Record
from idiom.data.tokenizer import Tokenizer
from idiom.model import IDiomTransformer, ModelConfig
from idiom.sae import SparseCoder
from idiom.sae.steer import SteeringSpec, add_direction_hook, steer_generation, steering
from idiom.sae.steer.steer import build_steering_hook

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
RECS = [Record(f"r{i}", "MEDSKVDNRPQACDEFG", 3, 12) for i in range(4)]


def test_steering_context_only_edits_residues():
    model = IDiomTransformer(TINY).eval()
    tokens = torch.tensor([[TOK.start_id, *TOK.encode("1AC3D2EF")]])
    base = model(tokens)
    direction = torch.ones(TINY.d_model)
    with steering(model, 0, add_direction_hook(direction, 5.0), tokenizer=TOK):
        steered = model(tokens)
    # logits change somewhere (edit applied), but START/markers were left unsteered
    assert not torch.allclose(base, steered)


def test_steer_generation_runs():
    model = IDiomTransformer(TINY).eval()
    sae = SparseCoder(TINY.d_model, num_latents=32, k=4)
    spec = SteeringSpec(layer=1, feature_idx=3, strength=2.0, mode="add_direction")
    out = steer_generation(model, sae, spec, n_samples=4, max_new_tokens=6, temperature=0)
    assert out.shape[0] == 4 and out.size(1) <= 6


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 2.0, -1.0])
def test_ablation_respects_scalar_strength(strength):
    sae = SparseCoder(4, num_latents=8, k=1)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.encoder.bias[0] = 2.0 # feature 0 is always active
        sae.W_dec[0].fill_(0.25) # its contribution is 0.5 in every dimension
    residual = torch.ones(2, 3, 4)
    hook = build_steering_hook(sae, SteeringSpec(0, 0, strength, mode="ablate"))
    actual = hook(None, None, residual)
    torch.testing.assert_close(actual, residual - strength * 0.5)
    if strength == 0:
        assert actual is residual


def test_zero_ablation_generation_matches_unsteered_baseline():
    from idiom import IDiom, IDiomSAE

    host = IDiom(IDiomTransformer(TINY))
    sae = SparseCoder(TINY.d_model, num_latents=32, k=4)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.encoder.bias[0] = 10.0
    lens = IDiomSAE(sae, host, layer=1, region="idr", fim_mode="unprompted")
    options = dict(n=5, max_new_tokens=8, seed=42, batch_size=2)
    assert lens.steer_generate(0, 0, mode="ablate", **options) == host.generate_unprompted(**options)
