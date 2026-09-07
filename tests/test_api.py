"""Public-API tests (CPU-only): save/from_pretrained round-trip + generation + embeddings."""

import torch

from idiom import IDiom, IDiomSAE, ModelConfig
from idiom.data.tokenizer import RESIDUES
from idiom.model import IDiomTransformer

TINY = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)


def _idiom():
    return IDiom(IDiomTransformer(TINY))


def test_save_and_from_pretrained_roundtrip(tmp_path):
    m = _idiom()
    m.save_pretrained(tmp_path / "rel")
    assert (tmp_path / "rel" / "config.json").exists()
    assert (tmp_path / "rel" / "model.safetensors").exists()

    loaded = IDiom.from_pretrained(tmp_path / "rel")
    tokens = torch.randint(0, TINY.vocab_size, (1, 6))
    assert torch.allclose(m.model(tokens), loaded.model(tokens), atol=1e-5)


def test_generate_unprompted_returns_residue_strings():
    seqs = _idiom().generate_unprompted(n=3, max_new_tokens=8, temperature=0, seed=0)
    assert len(seqs) == 3
    assert all(set(s) <= set(RESIDUES) for s in seqs)  # only residue chars (markers/controls stripped)


def test_generate_prompted_and_fasta(tmp_path):
    m = _idiom()
    seqs = m.generate_prompted("MEDSKVDNRPQ", 4, 8, n=2, max_new_tokens=6, temperature=0)
    assert len(seqs) == 2

    in_fa = tmp_path / "in.fasta"
    in_fa.write_text(">A_IDR_4-8\nMEDSKVDNRPQ\n")  # 1-based header -> internal half-open [3, 8)
    # Sample (temperature>0) so the random-init model emits non-empty IDRs (greedy decodes STOP
    # first -> empty). The writer drops empty generations, so the record count equals the same-seed
    # in-memory non-empty count, generated with the coords (3, 8) the writer uses internally.
    kw = dict(n=8, max_new_tokens=6, temperature=1.0, seed=0)
    expected = sum(bool(s) for s in m.generate_prompted("MEDSKVDNRPQ", 3, 8, **kw))
    assert expected > 0
    out = m.generate_prompted_fasta(in_fa, tmp_path / "out.fasta", **kw)
    assert out.read_text().count(">A_idiom_prompted_gen") == expected


def test_generate_cli(tmp_path):
    from idiom.api import main

    _idiom().save_pretrained(tmp_path / "rel")
    out = tmp_path / "idps.fasta"
    # Sample with a fixed seed (greedy would emit STOP first -> empty IDRs, all dropped by the
    # writer); compare against the same-seed in-memory non-empty count from the reloaded model.
    m = IDiom.from_pretrained(tmp_path / "rel")
    expected = sum(bool(s) for s in m.generate_unprompted(n=8, max_new_tokens=6, temperature=1.0, seed=0))
    assert expected > 0
    main(["unprompted", "--model", str(tmp_path / "rel"), "--out", str(out), "--n", "8",
          "--max-new-tokens", "6", "--temperature", "1.0", "--seed", "0"])
    assert out.read_text().count(">idiom_unprompted_") == expected


def test_embed(tmp_path):
    fa = tmp_path / "p.fasta"
    fa.write_text(">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n")
    emb = _idiom().embed(fa, layers=[1], pool="mean")
    values, index = emb[1]
    assert values.shape == (1, TINY.d_model) and index[0]["accession"] == "A"


def test_embed_plain_string_and_list():
    m = _idiom()
    # a bare sequence is treated as an unprompted IDR (the whole sequence is the IDR)
    values, index = m.embed("MEDSKVDNRPQACDEFG", layers=[1], pool="mean")[1]
    assert values.shape == (1, TINY.d_model) and index[0]["accession"] == "seq_0"
    # a list of bare sequences -> one row each, with synthetic accessions
    v2, idx2 = m.embed(["MEDSKVDN", "ACDEFGHIKL"], layers=[1], pool="mean")[1]
    assert v2.shape == (2, TINY.d_model) and [r["accession"] for r in idx2] == ["seq_0", "seq_1"]


def test_embed_noncanonical_raises():
    import pytest

    with pytest.raises(ValueError, match="canonical"):
        _idiom().embed("MEDSX", layers=[1])


def _idiom_sae(host, *, region="all", fim_mode="prompted"):
    from idiom.sae import SparseCoder

    sae = SparseCoder(TINY.d_model, num_latents=TINY.d_model * 4, k=8)
    return IDiomSAE(sae, host, layer=1, region=region, fim_mode=fim_mode)


def test_idiomsae_save_and_from_pretrained_roundtrip(tmp_path):
    host = _idiom()
    sae = _idiom_sae(host)
    host.save_pretrained(tmp_path / "rel")  # so the recorded host_model can be auto-loaded
    sdir = tmp_path / "sae_rel"
    sae.save_pretrained(sdir, host_model=str(tmp_path / "rel"))
    assert (sdir / "sae_config.json").exists() and (sdir / "sae.safetensors").exists()

    loaded = IDiomSAE.from_pretrained(sdir)  # no model= -> host auto-loaded from sae_config
    assert loaded.layer == 1
    x = torch.randn(5, TINY.d_model)
    assert torch.allclose(sae.sae.encode_dense(x), loaded.sae.encode_dense(x), atol=1e-5)


def test_idiomsae_encode_and_steer(tmp_path):
    host = _idiom()
    sae = _idiom_sae(host)
    fa = tmp_path / "p.fasta"
    fa.write_text(">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n")
    feats, accs = sae.encode(fa, pool="mean")
    assert feats.shape == (1, sae.sae.num_latents) and accs == ["A"]
    seqs = sae.steer_generate(feature=0, strength=1.0, n=2, max_new_tokens=6, temperature=0)
    assert len(seqs) == 2 and all(isinstance(s, str) for s in seqs)


def test_idiomsae_encode_rejects_a_region_an_unprompted_sae_cannot_produce():
    # an unprompted-mode SAE only ever sees "132{IDR}": there are no flanking residues to select,
    # so asking for them must say so rather than silently returning IDR features (or nothing)
    import pytest

    sae = _idiom_sae(_idiom(), fim_mode="unprompted")
    for region in ("all", "non_idr"):
        with pytest.raises(ValueError, match="trained in unprompted mode"):
            sae.encode(["MEDSKVDN"], pool="mean", region=region)
    feats, _ = sae.encode(["MEDSKVDN"], pool="mean", region="idr")  # the one it can produce
    assert feats.shape == (1, sae.sae.num_latents)


def test_idiomsae_repr_shows_the_training_distribution():
    # printing the object is the cheapest way for a downstream user to see the regime
    r = repr(_idiom_sae(_idiom(), region="idr", fim_mode="unprompted"))
    assert "region='idr'" in r and "fim_mode='unprompted'" in r and "layer=1" in r


def test_idiomsae_encode_plain_strings():
    sae = _idiom_sae(_idiom())
    feats, accs = sae.encode(["MEDSKVDN", "ACDEFGHIKL"], pool="mean")
    assert feats.shape == (2, sae.sae.num_latents) and accs == ["seq_0", "seq_1"]


def test_idiomsae_save_records_published_host_model(tmp_path):
    # at publish time the SAE's recorded host_model must become the Hub repo id, so a released
    # SAE can self-load its host; save_pretrained(host_model=...) is what rewrites it.
    import json

    host = _idiom()
    sae = _idiom_sae(host)
    sdir = sae.save_pretrained(tmp_path / "sae_rel", host_model="jxliu2/idiom-300M")
    cfg = json.loads((sdir / "sae_config.json").read_text())
    assert cfg["host_model"] == "jxliu2/idiom-300M"
    assert hasattr(sae, "push_to_hub")


def test_prompted_defaults_and_fasta(tmp_path):
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
