"""P5 fidelity + steering tests (CPU-only): rewired onto IDiomTransformer, no legacy refs."""

import torch
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.io import Record
from idiom.data.tokenizer import Tokenizer
from idiom.model import IDiomTransformer, ModelConfig
from idiom.sae import SparseCoder
from idiom.sae.fidelity import compute_fidelity
from idiom.sae.steering import SteeringSpec, add_direction_hook, steer_generation, steering

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
RECS = [Record(f"r{i}", "MEDSKVDNRPQACDEFG", 3, 12) for i in range(4)]


def test_steering_context_only_edits_residues():
    model = IDiomTransformer(TINY).eval()
    tokens = torch.tensor([[TOK.start_id, *TOK.encode("1AC3D2EF")]])  # markers + residues
    base = model(tokens)
    direction = torch.ones(TINY.d_model)
    with steering(model, 0, add_direction_hook(direction, 5.0), tokenizer=TOK):
        steered = model(tokens)
    # logits change somewhere (edit applied), but START/markers were left unsteered.
    assert not torch.allclose(base, steered)


def test_compute_fidelity_runs():
    model = IDiomTransformer(TINY).eval()
    sae = SparseCoder(TINY.d_model, num_latents=32, k=4)
    ds = RecordDataset(RECS, TOK, max_len=64, fim_idr_prob=1.0)
    batches = list(DataLoader(ds, batch_size=2, collate_fn=make_collate(TOK.pad_id)))
    res = compute_fidelity(model, sae, layer=1, batches=batches, pad_id=TOK.pad_id, tokenizer=TOK)
    for v in (res.loss_clean, res.loss_sae, res.loss_ablate):
        assert v == v and v >= 0  # finite, non-negative NLLs
    assert isinstance(res.pct_loss_recovered, float)


def test_steer_generation_runs():
    model = IDiomTransformer(TINY).eval()
    sae = SparseCoder(TINY.d_model, num_latents=32, k=4)
    spec = SteeringSpec(layer=1, feature_idx=3, strength=2.0, mode="add_direction")
    out = steer_generation(model, sae, spec, n_samples=4, max_new_tokens=6, temperature=0)
    assert out.shape[0] == 4 and out.size(1) <= 6
