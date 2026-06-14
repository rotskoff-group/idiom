"""P5 SAE entrypoint tests (CPU-only): pretrained loader + build() wiring."""

import torch
from omegaconf import OmegaConf

from idiom.model import IDiomTransformer, ModelConfig
from idiom.model.io import load_pretrained
from idiom.sae.train_sae import build

CFG = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)


def _save_ckpt(tmp_path):
    m = IDiomTransformer(CFG)
    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in m.state_dict().items()}}, ckpt)
    return m, ckpt


def test_load_pretrained_roundtrip(tmp_path):
    m, ckpt = _save_ckpt(tmp_path)
    loaded = load_pretrained(ckpt, CFG)
    for (_, a), (_, b) in zip(m.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b)
    assert not loaded.training  # loaded in eval mode


def test_sae_build_wires_and_streams(tmp_path):
    _, ckpt = _save_ckpt(tmp_path)
    fasta = tmp_path / "r.fasta"
    fasta.write_text(">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n>B_IDR_2-7\nACDEFGHIKLMN\n")
    cfg = OmegaConf.create({
        "seed": 0, "device": "cpu", "model_ckpt": str(ckpt),
        "model": {"n_layers": 2, "d_model": 16, "n_heads": 4, "max_seq_len": 64, "vocab_size": 27},
        "layer": 1,
        "data": {"fasta": str(fasta), "fim_full_prob": 1.0, "record_batch_size": 2},
        "sae_batch_size": 8, "buffer_size": 8, "init_b_dec_from_mean": True,
        "sae": {"k": 4, "expansion_factor": 2, "activation": "topk", "multi_topk": False,
                "auxk_alpha": 0.0, "dead_feature_tokens": 1000000, "warmup_steps": 1},
        "trainer": {"max_steps": 10},
    })
    lit, store = build(cfg)
    assert lit.sae.d_in == 16 and store.layer == 1
    batch = next(iter(store))  # streamed activations flow through the wired store
    assert batch.shape[1] == 16
