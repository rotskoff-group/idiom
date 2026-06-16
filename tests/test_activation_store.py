"""P5 SAE-streaming tests (CPU-only): ActivationStore -> LitSAE step, no h5."""

import torch
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.io import Record
from idiom.data.tokenizer import Tokenizer
from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.activations import extract_activations
from idiom.sae import SparseCoder
from idiom.sae.activation_store import ActivationStore
from idiom.sae.lit_sae import LitSAE

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
RECS = [Record(f"r{i}", "MEDSKVDNRPQACDEFG", 3, 12) for i in range(8)]


def _store(sae_batch_size=8, buffer_size=8, layer=1):
    model = IDiomTransformer(TINY)
    ds = RecordDataset(RECS, TOK, max_len=64, fim_full_prob=1.0)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate(TOK.pad_id))
    return ActivationStore(model, loader, layer, sae_batch_size=sae_batch_size,
                           buffer_size=buffer_size, device="cpu")


def test_store_yields_dmodel_batches():
    store = _store(sae_batch_size=8)
    batch = next(iter(store))
    assert batch.shape == (8, TINY.d_model)  # [sae_batch_size, d_model]
    assert torch.isfinite(batch).all()


def test_mean_activation_shape():
    assert _store().mean_activation(max_batches=2).shape == (TINY.d_model,)


def test_region_split_idr_vs_non_idr():
    """idr + non_idr partition all residues; counts match the FIM span (9 IDR + 8 flank / seq)."""
    model = IDiomTransformer(TINY)
    ds = RecordDataset(RECS, TOK, max_len=64, fim_full_prob=1.0)  # every sample is 'full'
    x = next(iter(DataLoader(ds, batch_size=8, collate_fn=make_collate(TOK.pad_id))))[0]
    n_all = extract_activations(model, x, [1], tokenizer=TOK, region="all")[1].values.size(0)
    n_idr = extract_activations(model, x, [1], tokenizer=TOK, region="idr")[1].values.size(0)
    n_non = extract_activations(model, x, [1], tokenizer=TOK, region="non_idr")[1].values.size(0)
    assert n_idr + n_non == n_all
    # full_seq len 17, IDR=[3,12) -> 9 IDR residues, 8 flank residues per sequence, x8 sequences
    assert n_idr == 9 * 8 and n_non == 8 * 8


def test_region_on_denovo_132_format():
    """de-novo '132{IDR}' has no flanks: region=idr keeps all 9 IDR/seq, non_idr keeps none."""
    model = IDiomTransformer(TINY)
    ds = RecordDataset(RECS, TOK, max_len=64, fim_full_prob=0.0)  # every sample is '132'
    x = next(iter(DataLoader(ds, batch_size=8, collate_fn=make_collate(TOK.pad_id))))[0]
    n_idr = extract_activations(model, x, [1], tokenizer=TOK, region="idr")[1].values.size(0)
    n_non = extract_activations(model, x, [1], tokenizer=TOK, region="non_idr")[1].values.size(0)
    assert n_idr == 9 * 8 and n_non == 0


def test_lit_sae_step_on_streamed_acts():
    store = _store(sae_batch_size=8)
    batch = next(iter(store))
    lit = LitSAE(d_in=TINY.d_model, k=4, expansion_factor=2, auxk_alpha=0.0, total_steps=10, warmup_steps=1)
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss) and loss.requires_grad
    # the SAE reconstructs d_model-dim vectors
    assert isinstance(lit.sae, SparseCoder) and lit.sae.d_in == TINY.d_model
