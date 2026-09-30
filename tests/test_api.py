"""Public-API tests: save/from_pretrained round-trip + generation + embeddings."""

import torch

from idiom import IDiom, IDiomSAE, ModelConfig
from idiom.data.tokenizer import RESIDUES
from idiom.model import IDiomTransformer

TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)


def _idiom():
    """Wrap a tiny transformer in the public IDiom API."""
    return IDiom(IDiomTransformer(TINY))


def test_save_and_from_pretrained_roundtrip(tmp_path):
    """Verify that saving and reloading preserves model configuration and weights."""
    m = _idiom()
    m.save_pretrained(tmp_path / "rel")
    assert (tmp_path / "rel" / "config.json").exists()
    assert (tmp_path / "rel" / "model.safetensors").exists()

    loaded = IDiom.from_pretrained(tmp_path / "rel")
    tokens = torch.randint(0, TINY.vocab_size, (1, 6))
    assert torch.allclose(m.model(tokens), loaded.model(tokens), atol=1e-5)


def test_generate_unprompted_returns_residue_strings():
    """Verify that unprompted generation returns only residue strings."""
    seqs = _idiom().generate_unprompted(n=3, max_new_tokens=8, temperature=0, seed=0)
    assert len(seqs) == 3
    assert all(set(s) <= set(RESIDUES) for s in seqs)


def test_generate_prompted_and_fasta(tmp_path):
    """Verify prompted generation and export of nonempty completions to FASTA."""
    m = _idiom()
    seqs = m.generate_prompted("MEDSKVDNRPQ", 4, 8, n=2, max_new_tokens=6, temperature=0)
    assert len(seqs) == 2

    in_fa = tmp_path / "in.fasta"
    # Sample to avoid immediate greedy STOP; compare with the same-seed non-empty outputs
    in_fa.write_text(">A_IDR_4-8\nMEDSKVDNRPQ\n")
    kw = dict(n=8, max_new_tokens=6, temperature=1.0, seed=0)
    expected = sum(bool(s) for s in m.generate_prompted("MEDSKVDNRPQ", 3, 8, **kw))
    assert expected > 0
    out = m.generate_prompted_fasta(in_fa, tmp_path / "out.fasta", **kw)
    assert out.read_text().count(">A_idiom_prompted_gen") == expected


def test_generate_cli(tmp_path):
    """Verify that the generation CLI writes the requested FASTA outputs."""
    from idiom.api import main

    _idiom().save_pretrained(tmp_path / "rel")
    out = tmp_path / "idps.fasta"
    # Sample to avoid immediate greedy STOP after reloading
    m = IDiom.from_pretrained(tmp_path / "rel")
    expected = sum(bool(s) for s in m.generate_unprompted(n=8, max_new_tokens=6, temperature=1.0, seed=0))
    assert expected > 0
    main(
        [
            "unprompted",
            "--model",
            str(tmp_path / "rel"),
            "--out",
            str(out),
            "--n",
            "8",
            "--max-new-tokens",
            "6",
            "--temperature",
            "1.0",
            "--seed",
            "0",
        ]
    )
    assert out.read_text().count(">idiom_unprompted_") == expected


def test_embed(tmp_path):
    """Verify mean-pooled embedding shape and accession metadata."""
    fa = tmp_path / "p.fasta"
    fa.write_text(">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n")
    emb = _idiom().embed(fa, layers=[1], pool="mean")
    values, index = emb[1]
    assert values.shape == (1, TINY.d_model) and index[0]["accession"] == "A"


def test_embed_plain_string_and_list():
    """Verify embeddings and assigned accessions for one sequence and a sequence list."""
    m = _idiom()
    values, index = m.embed("MEDSKVDNRPQACDEFG", layers=[1], pool="mean")[1]
    assert values.shape == (1, TINY.d_model) and index[0]["accession"] == "seq_0"
    v2, idx2 = m.embed(["MEDSKVDN", "ACDEFGHIKL"], layers=[1], pool="mean")[1]
    assert v2.shape == (2, TINY.d_model) and [r["accession"] for r in idx2] == ["seq_0", "seq_1"]


def test_embed_noncanonical_raises():
    """Verify that embedding rejects noncanonical sequences."""
    import pytest

    with pytest.raises(ValueError, match="canonical"):
        _idiom().embed("MEDSX", layers=[1])


def test_embed_long_bare_sequence():
    """Verify that long bare sequences embed identically to single-item lists."""
    cfg = ModelConfig(n_layers=1, d_model=16, n_heads=2, max_seq_len=512)
    model = IDiom(IDiomTransformer(cfg))
    seq = "ACDEFGHIKLMNPQRSTVWY" * 15
    bare, _ = model.embed(seq, layers=[0])[0]
    listed, _ = model.embed([seq], layers=[0])[0]
    assert (bare == listed).all()


def _idiom_sae(host, *, region="all", fim_mode="prompted"):
    """Attach a small SAE to the supplied host with the requested training distribution."""
    from idiom.sae import SparseCoder

    sae = SparseCoder(TINY.d_model, num_latents=TINY.d_model * 4, k=8)
    return IDiomSAE(sae, host, layer=1, region=region, fim_mode=fim_mode)


def test_idiomsae_save_and_from_pretrained_roundtrip(tmp_path):
    """Verify that SAE export and reload preserve metadata and feature activations."""
    host = _idiom()
    sae = _idiom_sae(host)
    host.save_pretrained(tmp_path / "rel")
    sdir = tmp_path / "sae_rel"
    sae.save_pretrained(sdir, host_model=str(tmp_path / "rel"))
    assert (sdir / "sae_config.json").exists() and (sdir / "sae.safetensors").exists()
    assert (sdir / "config.json").read_bytes() == (sdir / "sae_config.json").read_bytes()

    loaded = IDiomSAE.from_pretrained(sdir)
    assert loaded.layer == 1
    x = torch.randn(5, TINY.d_model)
    assert torch.allclose(sae.sae.encode_dense(x), loaded.sae.encode_dense(x), atol=1e-5)


def test_idiomsae_encode_and_steer(tmp_path):
    """Verify pooled SAE features and generation with feature steering."""
    host = _idiom()
    sae = _idiom_sae(host)
    fa = tmp_path / "p.fasta"
    fa.write_text(">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n")
    feats, accs = sae.encode(fa, pool="mean")
    assert feats.shape == (1, sae.sae.num_latents) and accs == ["A"]
    seqs = sae.steer_generate(feature=0, strength=1.0, n=2, max_new_tokens=6, temperature=0)
    assert len(seqs) == 2 and all(isinstance(s, str) for s in seqs)


def test_idiomsae_unprompted_encodes_only_idrs():
    """Full protein inputs use only their marked IDRs for an unprompted SAE."""
    from idiom.data.records import Record

    sae = _idiom_sae(_idiom(), region="idr", fim_mode="unprompted")
    feats, index = sae.encode(Record("p", "MEDQSSGACDE", 3, 7), pool="none")
    bare, _ = sae.encode("QSSG", pool="none")
    assert (feats == bare).all()
    assert [r["source_pos"] for r in index] == [3, 4, 5, 6]


def test_idiomsae_repr_shows_the_training_distribution():
    """Verify that the SAE representation includes region, FIM mode, and layer."""
    r = repr(_idiom_sae(_idiom(), region="idr", fim_mode="unprompted"))
    assert "region='idr'" in r and "fim_mode='unprompted'" in r and "layer=1" in r


def test_idiomsae_encode_plain_strings():
    """Verify that SAE encoding accepts a list of bare sequences."""
    sae = _idiom_sae(_idiom())
    feats, accs = sae.encode(["MEDSKVDN", "ACDEFGHIKL"], pool="mean")
    assert feats.shape == (2, sae.sae.num_latents) and accs == ["seq_0", "seq_1"]


def test_sae_keeps_multiple_idrs_of_one_protein_separate(tmp_path):
    """Verify that repeated accessions retain separate IDR records and feature rows."""
    import numpy as np

    from idiom.data.records import read_records

    fasta = tmp_path / "repeated.fasta"
    fasta.write_text(">P1_IDR_1-3\nACDEFGHIK\n>P1_IDR_6-9\nACDEFGHIK\n")
    records = list(read_records(fasta))
    for fim_mode in ("prompted", "unprompted"):
        sae = _idiom_sae(_idiom(), region="idr", fim_mode=fim_mode)
        pooled, accs = sae.encode(fasta)
        expected = np.concatenate([sae.encode(record)[0] for record in records])
        assert accs == ["P1", "P1"]
        np.testing.assert_allclose(pooled, expected, atol=1e-6)
        feats, index = sae.encode(fasta, pool="none")
        assert {row["record_idx"] for row in index} == {0, 1}
        groups = ["".join(row["residue"] for row in index if row["record_idx"] == i) for i in (0, 1)]
        assert groups == ["ACD", "GHIK"]


def test_idiomsae_save_records_published_host_model(tmp_path):
    """Verify that SAE export records the published host-model identifier."""
    import json

    host = _idiom()
    sae = _idiom_sae(host)
    sdir = sae.save_pretrained(tmp_path / "sae_rel", host_model="jxliu2/idiom-300M")
    cfg = json.loads((sdir / "sae_config.json").read_text())
    assert cfg["host_model"] == "jxliu2/idiom-300M"
    assert hasattr(sae, "push_to_hub")


def test_prompted_defaults_and_fasta(tmp_path):
    """Verify default prompted generation and FASTA export at the context limit."""
    import pytest

    model = _idiom()
    with pytest.warns(UserWarning, match="remaining context"):
        assert len(model.generate_prompted("ACDEFG", 1, 3, n=1)) == 1
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">A_IDR_2-3\nACDEFG\n")
    with pytest.warns(UserWarning, match="remaining context"):
        out = model.generate_prompted_fasta(fasta, tmp_path / "out.fasta", n=1, return_full=True)
    assert out.exists()


def test_generation_batches_are_bounded_and_repeatable():
    """Verify batch-size limits and repeatable generation with a fixed seed."""
    model = _idiom()
    sae = _idiom_sae(model)
    for generate in (model.generate_unprompted, lambda **kw: sae.steer_generate(0, 0.5, **kw)):
        sizes = []
        handle = model.model.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[0]))
        first = generate(n=19, max_new_tokens=3, seed=42)
        handle.remove()
        assert len(first) == 19
        assert max(sizes) == 8 and 3 in sizes
        assert first == generate(n=19, max_new_tokens=3, seed=42)
        sizes = []
        handle = model.model.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[0]))
        assert len(generate(n=5, batch_size=2, max_new_tokens=3, seed=42)) == 5
        handle.remove()
        assert max(sizes) == 2
