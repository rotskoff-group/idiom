"""Embedding alignment and SAE compatibility against direct model activations."""

import numpy as np
import pytest
import torch

from idiom import IDiom, IDiomSAE
from idiom.data.fim import fim_prompted, fim_unprompted, residue_source_positions
from idiom.data.io import Record
from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.extract import extract_embeddings, write_embeddings
from idiom.sae import SparseCoder

CFG = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)
RECORDS = [Record("same", "MEDQSSGACDE", 3, 7), Record("same", "ACDEFGHIK", 1, 4)]


def direct_rows(host, record, mode):
    """Read residue vectors directly from the transformer, bypassing extraction helpers."""
    build = fim_prompted if mode == "prompted" else fim_unprompted
    tokens = torch.tensor(
        [[host.tok.start_id, *host.tok.encode(build(record.full_seq, record.idr_start, record.idr_end))]]
    )
    with torch.no_grad():
        _, hidden = host.model(tokens, return_hidden_states=True)
    return hidden[1][0, tokens[0] < host.tok.n_residues]


@pytest.mark.parametrize("region", ["idr", "non_idr", "all"])
@pytest.mark.parametrize("order", ["sequence", "fim"])
def test_embeddings_match_direct_model_rows(region, order):
    host = IDiom(IDiomTransformer(CFG).eval())
    values, index = host.embed(RECORDS, [1], pool="none", region=region, order=order)[1]
    expected = []
    for ri, record in enumerate(RECORDS):
        src = residue_source_positions(len(record.full_seq), record.idr_start, record.idr_end, "prompted")
        selected = [
            i
            for i, p in enumerate(src)
            if region == "all" or ((record.idr_start <= p < record.idr_end) == (region == "idr"))
        ]
        if order == "sequence":
            selected.sort(key=lambda i: src[i])
        expected.append(direct_rows(host, record, "prompted")[selected].numpy())
        rows = [r for r in index if r["record_idx"] == ri]
        assert [r["source_pos"] for r in rows] == [src[i] for i in selected]
        assert [r["residue"] for r in rows] == [record.full_seq[src[i]] for i in selected]
    np.testing.assert_array_equal(values, np.concatenate(expected))
    pooled, meta = host.embed(RECORDS, [1], region=region)[1]
    np.testing.assert_allclose(pooled, np.stack([x.mean(0) for x in expected]), atol=1e-7)
    assert [r["n_residues"] for r in meta] == [len(x) for x in expected]


@pytest.mark.parametrize("activation", ["topk", "groupmax"])
@pytest.mark.parametrize(
    "mode,region", [("prompted", "idr"), ("prompted", "non_idr"), ("prompted", "all"), ("unprompted", "idr")]
)
def test_sae_preserves_legacy_features_pooling_and_order(mode, region, activation):
    host = IDiom(IDiomTransformer(CFG).eval())
    coder = SparseCoder(CFG.d_model, num_latents=64, k=8, activation=activation)
    lens = IDiomSAE(coder, host, layer=1, region=region, fim_mode=mode)
    # Independent reference reproduces the original encode-then-pool computation.
    raw = [direct_rows(host, rec, mode) for rec in RECORDS]
    with torch.no_grad():
        expected = coder.encode_dense(torch.cat(raw)).numpy()
    actual, index = lens.encode(RECORDS, pool="none")
    np.testing.assert_array_equal(actual, expected)
    pooled_expected = []
    offset = 0
    for ri, rec in enumerate(RECORDS):
        src = residue_source_positions(len(rec.full_seq), rec.idr_start, rec.idr_end, mode)
        rows = [r for r in index if r["record_idx"] == ri]
        assert [r["source_pos"] for r in rows] == src
        selected = [
            offset + i
            for i, p in enumerate(src)
            if region == "all" or ((rec.idr_start <= p < rec.idr_end) == (region == "idr"))
        ]
        pooled_expected.append(expected[selected].mean(0))
        offset += len(src)
    for order in ("fim", "sequence"):
        pooled, accs = lens.encode(RECORDS, order=order)
        assert accs == ["same", "same"]
        np.testing.assert_array_equal(pooled, np.stack(pooled_expected))
    reordered, metadata = lens.encode(RECORDS, pool="none", order="sequence")
    permutation = sorted(range(len(index)), key=lambda i: (index[i]["record_idx"], index[i]["source_pos"]))
    np.testing.assert_array_equal(reordered, expected[permutation])
    assert metadata == [index[i] for i in permutation]
    # Explicit region keeps its historical meaning for per-residue SAE output.
    explicit, _ = lens.encode(RECORDS, pool="none", region=region)
    np.testing.assert_array_equal(explicit, expected)


def test_empty_selection(tmp_path):
    host = IDiom(IDiomTransformer(CFG).eval())
    values, index = host.embed("ACDE", [1], pool="none", region="non_idr")[1]
    assert values.shape == (0, CFG.d_model) and index == []
    write_embeddings({1: (values, index)}, tmp_path)
    assert np.load(tmp_path / "layer_1.npy").shape == values.shape
    with pytest.raises(ValueError, match="no residues"):
        host.embed("ACDE", [1], region="non_idr")


@pytest.mark.parametrize("options", [{"pool": "bad"}, {"region": "bad"}, {"order": "bad"}])
def test_invalid_embedding_options(options):
    with pytest.raises(ValueError):
        extract_embeddings(IDiomTransformer(CFG).eval(), RECORDS, [1], **options)


def test_sae_preserves_empty_region_pooling():
    host = IDiom(IDiomTransformer(CFG).eval())
    lens = IDiomSAE(SparseCoder(CFG.d_model, num_latents=64, k=8), host, layer=1, region="non_idr")
    feats, accs = lens.encode("ACDE")
    assert feats.shape == (0, 64) and accs == []
    feats, index = lens.encode("ACDE", pool="none")
    assert feats.shape == (4, 64) and len(index) == 4
    with pytest.raises(RuntimeError, match="no residue rows"):
        lens.encode([])


def test_cli_region_and_order(tmp_path):
    from idiom.model.extract import main

    host = IDiom(IDiomTransformer(CFG).eval())
    release = host.save_pretrained(tmp_path / "model")
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">P_IDR_4-7\nMEDQSSGACDE\n")
    output = tmp_path / "output"
    main(
        [
            "--model",
            str(release),
            "--fasta",
            str(fasta),
            "--layers",
            "1",
            "--pool",
            "none",
            "--region",
            "all",
            "--order",
            "sequence",
            "--out",
            str(output),
        ]
    )
    values, _ = host.embed(fasta, [1], pool="none", region="all")[1]
    np.testing.assert_array_equal(np.load(output / "layer_1.npy"), values)
