"""Embedding alignment and SAE compatibility against direct model activations."""

import numpy as np
import pytest
import torch

from idiom import IDiom, IDiomSAE
from idiom.data.fim import fim_prompted, fim_unprompted, residue_source_positions
from idiom.data.io import Record
from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.extract import extract_embeddings
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


@pytest.mark.parametrize("mode", ["prompted", "unprompted"])
def test_idr_embeddings_and_pooling_match_direct_model(mode):
    host = IDiom(IDiomTransformer(CFG).eval())
    values, index = extract_embeddings(host.model, RECORDS, [1], pool="none", fim_mode=mode)[1]
    expected = []
    for ri, record in enumerate(RECORDS):
        src = residue_source_positions(len(record.full_seq), record.idr_start, record.idr_end, mode)
        selected = [i for i, p in enumerate(src) if record.idr_start <= p < record.idr_end]
        expected.append(direct_rows(host, record, mode)[selected].numpy())
        rows = [r for r in index if r["record_idx"] == ri]
        assert [r["source_pos"] for r in rows] == list(range(record.idr_start, record.idr_end))
        assert "".join(r["residue"] for r in rows) == record.full_seq[record.idr_start : record.idr_end]
        assert all(r["is_idr"] for r in rows)
    np.testing.assert_array_equal(values, np.concatenate(expected))
    for pool in ("mean", "last"):
        pooled, meta = extract_embeddings(host.model, RECORDS, [1], pool=pool, fim_mode=mode)[1]
        reference = [x.mean(0) if pool == "mean" else x[-1] for x in expected]
        np.testing.assert_allclose(pooled, np.stack(reference), atol=1e-7)
        assert [r["n_idr"] for r in meta] == [len(x) for x in expected]
    if mode == "prompted":
        public, _ = host.embed(RECORDS, [1], pool="none")[1]
        np.testing.assert_array_equal(public, values)
        # Removing flanks changes the causal representations, despite IDR-only output.
        bare, _ = host.embed("QSSG", [1], pool="none")[1]
        assert not np.allclose(bare, expected[0])


@pytest.mark.parametrize("activation", ["topk", "groupmax"])
@pytest.mark.parametrize("mode,region", [("prompted", "idr"), ("prompted", "all"), ("unprompted", "idr")])
def test_sae_idr_features_and_pooling(mode, region, activation, tmp_path):
    from idiom.sae.features.enrichment import feature_counts

    host = IDiom(IDiomTransformer(CFG).eval())
    coder = SparseCoder(CFG.d_model, num_latents=64, k=8, activation=activation)
    lens = IDiomSAE(coder, host, layer=1, region=region, fim_mode=mode)
    # Encode all original residue activations before selecting the IDR reference.
    raw = [direct_rows(host, rec, mode) for rec in RECORDS]
    with torch.no_grad():
        original = coder.encode_dense(torch.cat(raw)).numpy()
    expected = []
    offset = 0
    for rec, x in zip(RECORDS, raw):
        src = residue_source_positions(len(rec.full_seq), rec.idr_start, rec.idr_end, mode)
        selected = [offset + i for i, p in enumerate(src) if rec.idr_start <= p < rec.idr_end]
        expected.append(original[selected])
        offset += len(x)
    actual, index = lens.encode(RECORDS, pool="none")
    np.testing.assert_allclose(actual, np.concatenate(expected), atol=1e-7)
    assert all(r["is_idr"] for r in index)
    for pool in ("mean", "max"):
        pooled, accs = lens.encode(RECORDS, pool=pool)
        assert accs == ["same", "same"]
        reference = [x.mean(0) if pool == "mean" else x.max(0) for x in expected]
        np.testing.assert_allclose(pooled, np.stack(reference), atol=1e-7)
    if mode == "unprompted":
        bare, _ = lens.encode([r.full_seq[r.idr_start : r.idr_end] for r in RECORDS], pool="none")
        np.testing.assert_array_equal(actual, bare)
    if region == "idr":
        directory = lens.build_feature_dataset(RECORDS, tmp_path / "features", batch_size=1)
        counts, n = feature_counts(directory)
        np.testing.assert_array_equal(counts, (pooled > 0).sum(0))
        assert n == 2


@pytest.mark.parametrize("pool", ["none", "mean", "last"])
def test_empty_idrs_rejected(pool):
    host = IDiom(IDiomTransformer(CFG).eval())
    with pytest.raises(ValueError, match="IDR span"):
        host.embed(Record("empty", "ACDE", 2, 2), [1], pool=pool)


def test_invalid_options_and_non_idr_sae():
    host = IDiom(IDiomTransformer(CFG).eval())
    lens = IDiomSAE(SparseCoder(CFG.d_model, num_latents=64, k=8), host, layer=1, region="non_idr")
    with pytest.raises(ValueError, match="non_idr"):
        lens.encode(RECORDS)
    with pytest.raises(ValueError, match="pool"):
        host.embed(RECORDS, [1], pool="max")
    with pytest.raises(ValueError, match="pool"):
        lens.encode(RECORDS, pool="last")
    for option in ({"region": "all"}, {"order": "fim"}):
        with pytest.raises(TypeError):
            host.embed(RECORDS, [1], **option)
        with pytest.raises(TypeError):
            lens.encode(RECORDS, **option)


@pytest.mark.parametrize("pool", ["none", "mean", "last"])
def test_cli_idr_pooling(tmp_path, pool):
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
            pool,
            "--out",
            str(output),
        ]
    )
    values, _ = host.embed(fasta, [1], pool=pool)[1]
    np.testing.assert_array_equal(np.load(output / "layer_1.npy"), values)
