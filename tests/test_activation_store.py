"""SAE-streaming tests: ActivationStore -> LitSAE step, and region partitioning."""

import torch
from torch.utils.data import DataLoader

from idiom.data.dataset import RecordDataset, make_collate
from idiom.data.io import Record
from idiom.data.tokenizer import Tokenizer
from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.activations import extract_activations
from idiom.sae import SparseCoder
from idiom.sae.train.activation_store import ActivationStore
from idiom.sae.train.lit_sae import LitSAE

TOK = Tokenizer()
TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
RECS = [Record(f"r{i}", "MEDSKVDNRPQACDEFG", 3, 12) for i in range(8)]


def _store(sae_batch_size=8, buffer_size=8, layer=1):
    """Build a CPU activation store backed by a tiny transformer and padded record batches."""
    model = IDiomTransformer(TINY)
    ds = RecordDataset(RECS, TOK, max_len=64, prompted_prob=1.0)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate(TOK.pad_id))
    return ActivationStore(
        model, loader, layer, sae_batch_size=sae_batch_size, buffer_size=buffer_size, device="cpu"
    )


def test_store_yields_dmodel_batches():
    """Verify that activation batches have the expected shape and finite values."""
    store = _store(sae_batch_size=8)
    batch = next(iter(store))
    assert batch.shape == (8, TINY.d_model)
    assert torch.isfinite(batch).all()


def test_store_exhausts_after_exact_drain_without_losing_rows():
    """Verify that an exact buffer drain preserves every activation row."""
    store = _store(sae_batch_size=8, buffer_size=8)
    expected = torch.cat([store._acts(store._input_tokens(b)) for b in store.record_loader])
    actual = torch.cat(list(store))
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.sort(dim=0).values, expected.sort(dim=0).values)


def test_empty_activation_stream_yields_nothing():
    """Verify that an empty activation stream produces no batches."""
    store = ActivationStore(IDiomTransformer(TINY), [], 1)
    assert list(store) == []


def test_mean_activation_shape():
    """Verify that the mean activation has one value per model dimension."""
    assert _store().mean_activation(max_batches=2).shape == (TINY.d_model,)


def test_region_split_idr_vs_non_idr():
    """Verify that IDR and flank selections partition the residue activations."""
    model = IDiomTransformer(TINY)
    ds = RecordDataset(RECS, TOK, max_len=64, prompted_prob=1.0)
    x = next(iter(DataLoader(ds, batch_size=8, collate_fn=make_collate(TOK.pad_id))))[0]
    n_all = extract_activations(model, x, [1], tokenizer=TOK, region="all")[1].values.size(0)
    n_idr = extract_activations(model, x, [1], tokenizer=TOK, region="idr")[1].values.size(0)
    n_non = extract_activations(model, x, [1], tokenizer=TOK, region="non_idr")[1].values.size(0)
    assert n_idr + n_non == n_all
    # full_seq len 17, IDR=[3,12) -> 9 IDR residues, 8 flank residues per sequence, x8 sequences
    assert n_idr == 9 * 8 and n_non == 8 * 8


def test_region_on_unprompted_132_format():
    """Verify that unprompted inputs contain IDR activations and no flank activations."""
    model = IDiomTransformer(TINY)
    ds = RecordDataset(RECS, TOK, max_len=64, prompted_prob=0.0)
    x = next(iter(DataLoader(ds, batch_size=8, collate_fn=make_collate(TOK.pad_id))))[0]
    n_idr = extract_activations(model, x, [1], tokenizer=TOK, region="idr")[1].values.size(0)
    n_non = extract_activations(model, x, [1], tokenizer=TOK, region="non_idr")[1].values.size(0)
    assert n_idr == 9 * 8 and n_non == 0


def test_lit_sae_step_on_streamed_acts():
    """Verify that streamed activations produce a finite, differentiable SAE loss."""
    store = _store(sae_batch_size=8)
    batch = next(iter(store))
    lit = LitSAE(d_in=TINY.d_model, k=4, expansion_factor=2, auxk_alpha=0.0, total_steps=10, warmup_steps=1)
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss) and loss.requires_grad
    assert isinstance(lit.sae, SparseCoder) and lit.sae.d_in == TINY.d_model
