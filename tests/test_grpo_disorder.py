"""Disorder diagnostics: real V3 inference and optimizer-step logging semantics."""

import csv

import lightning as L
import numpy as np
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from idiom.train.grpo import LitGRPO
from idiom.utils.disorder import disorder_totals
from tests.reward_fixtures import proline_terms
from tests.test_grpo_module import TINY, TOK


def test_real_v3_batch_and_rng():
    before = torch.random.get_rng_state().clone()
    total, count, size = disorder_totals(["GSPQEKGSPQEK", "", "LLLLVVVVFFFF"])
    assert count == 2 and size == 3 and 0 < total < 2
    assert torch.equal(before, torch.random.get_rng_state())
    a, _, _ = disorder_totals(["GSPQEKGSPQEK"])
    b, _, _ = disorder_totals(["LLLLVVVVFFFF"])
    assert a > b
    assert total == pytest.approx(a + b, abs=1e-3)
    assert disorder_totals(["", ""]) == (0, 0, 2)


def test_disorder_averages_sequences_not_residues(monkeypatch):
    import metapredict

    monkeypatch.setattr(
        metapredict,
        "predict_disorder_batch",
        lambda seqs, **kw: [
            ["A", np.array([0.2])],
            ["GGG", np.array([0.6, 0.8, 1.0])],
        ],
    )
    assert disorder_totals(["A", "", "GGG"]) == pytest.approx((1.0, 2, 3))


@pytest.mark.parametrize("scores", [np.array([float("nan")]), np.array([1.2]), np.array([])])
def test_disorder_rejects_bad_predictions(monkeypatch, scores):
    import metapredict

    monkeypatch.setattr(metapredict, "predict_disorder_batch", lambda seqs, **kw: [["A", scores]])
    with pytest.raises(ValueError, match="metapredict returned"):
        disorder_totals(["A"])


@pytest.mark.parametrize("enabled", [True, False])
def test_logs_once_per_optimizer_step_with_accumulation(tmp_path, monkeypatch, enabled):
    import idiom.train.grpo.lit_grpo as module

    calls = []

    def score(seqs):
        calls.append(list(seqs))
        # Empty sequences excluded; A has mean disorder .2, GG has .8
        return sum({"": 0, "A": 0.2, "GG": 0.8}[s] for s in seqs), sum(bool(s) for s in seqs), len(seqs)

    monkeypatch.setattr(module, "disorder_totals", score)
    # Two sequences per microbatch; optimizer steps see 4, 4, then 2 sequences
    decoded = iter(["A", "", "GG", "GG", "", "", "", "", "A", "GG"])
    lit = LitGRPO(
        TINY, proline_terms(), group_size=2, max_new_tokens=3, log_samples_every=0, track_disorder=enabled
    )
    monkeypatch.setattr(lit, "_decode_idr", lambda completion: next(decoded))
    logger = CSVLogger(tmp_path, name="metrics")
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        max_steps=3,
        accumulate_grad_batches=2,
        log_every_n_steps=1,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    prompts = torch.tensor([TOK.encode("132")] * 5)
    trainer.fit(lit, train_dataloaders=DataLoader(prompts, batch_size=1))
    with open(f"{logger.log_dir}/metrics.csv") as f:
        rows = list(csv.DictReader(f))
    if enabled:
        assert [len(c) for c in calls] == [4, 4, 2]
        assert len(rows) == 3
        assert [float(r["train/metapredict_disorder"]) for r in rows] == pytest.approx([0.6, 0, 0.5])
        assert [float(r["train/metapredict_empty_fraction"]) for r in rows] == pytest.approx([0.25, 1, 0])
        assert [int(r["step"]) for r in rows] == [0, 1, 2]
        assert not lit._disorder_sequences
    else:
        assert not calls
        assert all("train/metapredict_disorder" not in row for row in rows)
