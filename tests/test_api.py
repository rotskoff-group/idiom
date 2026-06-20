"""Public-API tests (CPU-only): save/from_pretrained round-trip + generation + embeddings."""

import torch

from idiom import IDiom, IDiomSAE, ModelConfig
from idiom.model import IDiomTransformer
from idiom.data.tokenizer import RESIDUES

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


def test_generate_idp_returns_residue_strings():
    seqs = _idiom().generate_idp(n=3, max_new_tokens=8, temperature=0, seed=0)
    assert len(seqs) == 3
    assert all(set(s) <= set(RESIDUES) for s in seqs)  # only residue chars (markers/controls stripped)


def test_generate_idr_and_fasta(tmp_path):
    m = _idiom()
    seqs = m.generate_idr("MEDSKVDNRPQ", 4, 8, n=2, max_new_tokens=6, temperature=0)
    assert len(seqs) == 2

    in_fa = tmp_path / "in.fasta"
    in_fa.write_text(">A_IDR_4-8\nMEDSKVDNRPQ\n")  # 1-based header -> internal half-open
    out = m.generate_idr_fasta(in_fa, tmp_path / "out.fasta", n=2, max_new_tokens=6, temperature=0)
    text = out.read_text()
    assert text.count(">A_idiom_idr_gen") == 2


def test_generate_cli(tmp_path):
    from idiom.api import main

    _idiom().save_pretrained(tmp_path / "rel")
    out = tmp_path / "idps.fasta"
    main(["idp", "--model", str(tmp_path / "rel"), "--out", str(out), "--n", "2",
          "--max-new-tokens", "6", "--temperature", "0"])
    assert out.read_text().count(">idiom_idp_") == 2


def test_embed(tmp_path):
    fa = tmp_path / "p.fasta"
    fa.write_text(">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n")
    emb = _idiom().embed(fa, layers=[1], pool="mean")
    values, index = emb[1]
    assert values.shape == (1, TINY.d_model) and index[0]["accession"] == "A"


def _idiom_sae(host):
    from idiom.sae import SparseCoder

    sae = SparseCoder(TINY.d_model, num_latents=TINY.d_model * 4, k=8)
    return IDiomSAE(sae, host, layer=1)


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
