"""Execute the cookbook workflows with small real models and local FASTA inputs."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from idiom import IDiom, IDiomSAE, ModelConfig
from idiom.data.io import read_records
from idiom.model import IDiomTransformer
from idiom.sae import SparseCoder

NOTEBOOKS = Path(__file__).resolve().parents[1] / "cookbook" / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from workflow_utils import check_context, isolated, load_inputs, summaries, write_fasta  # noqa: E402


def test_input_audit_and_roundtrip(tmp_path):
    """Verify input auditing and sequence-file round trips in notebook helpers."""
    fasta = tmp_path / "input.fasta"
    fasta.write_text(
        ">same_IDR_2-4\nACDEFG\n>same_IDR_3-5\nACDEFG\n"
        ">bad_IDR_0-4\nACDEFG\n>missing\nACDEFG\n>ambiguous_IDR_1-3\nAXD\n"
    )
    records, audit = load_inputs(fasta, "annotated", None)
    assert len(records) == 2
    assert (audit.status == "accepted").tolist() == [True, True, False, False, False]
    assert audit.accession[:2].tolist() == ["same", "same"]
    assert records[0].accession != records[1].accession
    assert [r.full_seq for r in isolated(records)] == ["CDE", "DEF"]
    assert summaries(records, audit).idr_start_1based.tolist() == [2, 3]
    exported = write_fasta(records, tmp_path / "roundtrip.fasta")
    assert list(read_records(exported)) == records
    assert len(load_inputs(fasta, "idr", None)[0]) == 1
    limited, limit_audit = load_inputs(fasta, "annotated", 1)
    assert len(limited) == 1 and limit_audit.status.iloc[1] == "outside sample limit"
    with pytest.raises(ValueError, match="context"):
        check_context(records, 7, include_flanks=True)
    check_context(records, 7)
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    assert load_inputs(empty)[0] == []


@pytest.mark.parametrize(
    "name,use_flanks,pool",
    [
        ("analyze_sequences", False, "mean"),
        ("analyze_sequences", True, "mean"),
        ("analyze_sequences", False, "last"),
        ("analyze_sequences", True, "last"),
        ("generate_sequences", False, "mean"),
        ("inspect_sae_features", False, "mean"),
        ("feature_enrichment", False, "mean"),
        ("feature_enrichment_gallery", False, "mean"),
        ("compare_sequence_sets", False, "mean"),
        ("steer_generation", False, "mean"),
    ],
)
def test_notebook_execution(name, use_flanks, pool, tmp_path, monkeypatch):
    """Execute cookbook notebooks with local fixtures and verify their outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(NOTEBOOKS)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=128)
    host = IDiom(IDiomTransformer(config))
    sae = IDiomSAE(
        SparseCoder(16, num_latents=32, k=4),
        host,
        layer=1,
        host_model="test-host",
        region="idr",
        fim_mode="unprompted",
    )
    monkeypatch.setattr(IDiom, "from_pretrained", classmethod(lambda cls, *args, **kwargs: host))
    monkeypatch.setattr(IDiomSAE, "from_pretrained", classmethod(lambda cls, *args, **kwargs: sae))
    positive = tmp_path / "positive.fasta"
    background = tmp_path / "background.fasta"
    positive.write_text("".join(f">same_IDR_3-18\nAC{'Q' * (8 + i)}{'S' * (8 - i)}DE\n" for i in range(8)))
    background.write_text("".join(f">same_IDR_3-18\nAC{'E' * (8 + i)}{'K' * (8 - i)}DE\n" for i in range(8)))
    gallery = name == "feature_enrichment_gallery"
    if gallery:
        from idiom.sae.features import enrichment

        name = "feature_enrichment"
        # Exercise rendering/export even when random test weights yield no significant hits.
        monkeypatch.setattr(
            enrichment, "top_features", lambda result, **kwargs: [int(np.argmax(result["a"]))]
        )
    namespace = {"__name__": "__main__"}
    notebook = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text())
    try:
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            assert not cell["outputs"] and cell["execution_count"] is None
            exec(compile(source, f"{name}:cell_{i}", "exec"), namespace)
            if "OUT_DIR = Path(" in source:
                namespace.update(
                    OUT_DIR=tmp_path / "outputs",
                    DEVICE="cpu",
                    USE_FLANKS=use_flanks,
                    POOL=pool,
                    LAYER=1,
                    INPUT_FASTA=positive,
                    INPUT_MODE="annotated",
                    MAX_RECORDS=8,
                    N=4,
                    MAX_NEW_TOKENS=12,
                    LENGTH_RANGE=None,
                    FEATURE_ID=0,
                    POSITIVE_FASTA=positive,
                    POSITIVE_MODE="annotated",
                    BACKGROUND_FASTA=background,
                    BACKGROUND_MODE="annotated",
                    QUERY_FASTA=positive,
                    QUERY_MODE="annotated",
                    REFERENCE_FASTA=background,
                    REFERENCE_MODE="annotated",
                    RUN_SAE=True,
                    RUN_PERPLEXITY=True,
                    EXPORT_SIGNATURE=True,
                    MAX_POSITIVE=8,
                    MAX_BACKGROUND=8,
                )
        out = tmp_path / "outputs"
        run = json.loads((out / "run.json").read_text())
        assert run["elapsed_seconds"] > 0
        assert list(out.glob("*.csv")) and list(out.glob("*.png"))
        if name == "analyze_sequences":
            assert np.load(out / "embeddings.npy").shape == (8, 16)
            table = namespace["residue_table"]
            assert table.protein_position_1based.tolist() == list(range(3, 19))
            assert table.source_pos.tolist() == list(range(2, 18) if use_flanks else range(16))
            assert table.is_idr.all()
            assert "".join(table.residue) == "Q" * 8 + "S" * 8
            residue_values = np.load(out / "first_sequence_residue_embeddings.npy")
            assert residue_values.shape == (16, 16)
            expected = residue_values.mean(0) if pool == "mean" else residue_values[-1]
            np.testing.assert_allclose(np.load(out / "embeddings.npy")[0], expected, atol=1e-7)
            assert run["settings"]["pool"] == pool
            assert run["settings"]["use_flanks"] == use_flanks
        if name == "generate_sequences":
            redesigned = list(read_records(out / "redesigned_proteins.fasta"))
            assert redesigned and all(r.idr_start == 2 for r in redesigned)
            assert all(r.full_seq[:2] == "AC" and r.full_seq.endswith("DE") for r in redesigned)
        if name == "inspect_sae_features":
            assert namespace["trace_rows"]
            assert min(r["protein_position"] for r in namespace["trace_rows"]) == 3
        if name == "feature_enrichment":
            assert len(namespace["table"]) == 32
            assert set(map(lambda r: r.full_seq, namespace["positives"])).isdisjoint(
                r.full_seq for r in namespace["background"]
            )
        if gallery:
            assert (out / "signature.json").exists()
            assert (out / "feature_logos.png").exists()
        if name == "compare_sequence_sets":
            assert (out / "feature_prevalence.csv").exists()
            assert (out / "aggregate_perplexity.csv").exists()
    finally:
        plt.close("all")
        torch.set_num_threads(previous_threads)
