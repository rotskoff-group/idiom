"""P5 SAE entrypoint tests (CPU-only): build() wires a frozen model (arch read from ckpt)."""

from dataclasses import asdict

import torch
from omegaconf import OmegaConf

from idiom.model import IDiomTransformer, ModelConfig
from idiom.sae.training.train_sae import build

CFG = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=64)


def _save_ckpt(tmp_path):
    """Self-describing ckpt (carries its ModelConfig), like a real training checkpoint."""
    m = IDiomTransformer(CFG)
    ckpt = tmp_path / "m.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in m.state_dict().items()},
         "hyper_parameters": {"model_cfg": asdict(CFG)}},
        ckpt,
    )
    return m, ckpt


def test_sae_build_wires_and_streams(tmp_path):
    _, ckpt = _save_ckpt(tmp_path)
    fasta = tmp_path / "r.fasta"
    fasta.write_text(">A_IDR_3-9\nMEDSKVDNRPQACDEFG\n>B_IDR_2-7\nACDEFGHIKLMN\n")
    cfg = OmegaConf.create({
        "seed": 0, "device": "cpu", "model_ckpt": str(ckpt),  # arch read from the ckpt
        "layer": 1,
        "data": {"fasta": str(fasta), "fim_idr_prob": 1.0, "record_batch_size": 2},
        "sae_batch_size": 8, "buffer_size": 8, "init_b_dec_from_mean": True,
        "sae": {"k": 4, "expansion_factor": 2, "activation": "topk", "multi_topk": False,
                "auxk_alpha": 0.0, "dead_feature_tokens": 1000000, "warmup_steps": 1},
        "trainer": {"max_steps": 10},
    })
    lit, store = build(cfg)
    assert lit.sae.d_in == 16 and store.layer == 1
    batch = next(iter(store))  # streamed activations flow through the wired store
    assert batch.shape[1] == 16
