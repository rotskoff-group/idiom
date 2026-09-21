"""Execute the cookbook workflows with small real models and local FASTA inputs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from idiom import IDiom, IDiomSAE, ModelConfig
from idiom.data.records import read_records
from idiom.model import IDiomTransformer
from idiom.sae import SparseCoder
from idiom.utils.notebook_helpers import check_context, isolated, load_inputs, summaries, write_fasta

NOTEBOOKS = Path(__file__).resolve().parents[1] / "cookbook" / "notebooks"


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


NOTEBOOK_NAMES = [
    "01_generate_idrs",
    "02_explore_embeddings",
    "03_interpret_sae_features",
    "04_discover_feature_signature",
    "05_finetune_and_generate",
    "06_design_with_custom_rewards",
    "07_design_with_rl_sae",
]


def execute_notebook(name, parameters):
    """Execute all ordinary Python cells from a fresh namespace, replacing only user settings."""
    namespace = {"__name__": "__main__"}
    notebook = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text())
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell["outputs"] and cell["execution_count"] is None
        source = "".join(cell["source"])
        exec(compile(source, f"{name}:cell_{i}", "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(parameters)
    return namespace


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_execution(name, tmp_path, monkeypatch):
    """Run inference, training, checkpoint reload, and exports with real tiny models on CPU."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from idiom.sae.features import write_signature
    from idiom.train.grpo.reward import sae_feature

    monkeypatch.chdir(tmp_path)  # No checkout-relative files or helper imports
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config = ModelConfig(vocab_size=27, n_layers=2, d_model=16, n_heads=4, max_seq_len=128)
    host = IDiom(IDiomTransformer(config))
    model_dir = host.save_pretrained(tmp_path / "host")
    sae = IDiomSAE(
        SparseCoder(16, num_latents=32, k=4),
        host,
        layer=1,
        host_model=str(model_dir),
        region="idr",
        fim_mode="unprompted",
    )
    sae_dir = sae.save_pretrained(tmp_path / "sae")
    positive = tmp_path / "positive.fasta"
    background = tmp_path / "background.fasta"
    positive.write_text("".join(f">same_IDR_3-18\nAC{'Q' * (8 + i)}{'S' * (8 - i)}DE\n" for i in range(8)))
    background.write_text("".join(f">same_IDR_3-18\nAC{'E' * (8 + i)}{'K' * (8 - i)}DE\n" for i in range(8)))
    signature = write_signature(
        tmp_path / "targets.json", {"demo": [0, 1], "second": [1, 2]}, provenance={"sae": str(sae_dir)}
    )
    parameters = dict(
        OUT_DIR=tmp_path / "outputs",
        MODEL_ID=str(model_dir),
        SAE_ID=str(sae_dir),
        DEVICE="cpu",
        BATCH_SIZE=2,
        LAYER=1,
        INPUT_FASTA=positive,
        INPUT_MODE="annotated",
        MAX_RECORDS=8,
        N=4,
        MAX_NEW_TOKENS=12,
        LENGTH_RANGE=None,
        MAX_STEPS=2,
        GROUP_SIZE=2,
        TARGET_LENGTH=8,
        POSITIVE_FASTA=positive,
        POSITIVE_MODE="annotated",
        BACKGROUND_FASTA=background,
        BACKGROUND_MODE="annotated",
        MAX_POSITIVE=8,
        MAX_BACKGROUND=8,
        SIGNATURE_FILE=signature,
        SIGNATURE_NAMES=["demo", "second"],
        SAE_DEVICE="cpu",
    )
    try:
        namespace = execute_notebook(name, parameters)
        out = parameters["OUT_DIR"]
        assert json.loads((out / "run.json").read_text())["elapsed_seconds"] > 0
        assert list(out.glob("*.csv")) and list(out.glob("*.png"))
        assert out.with_suffix(".zip").exists()
        if name.startswith("01"):
            redesigned = list(read_records(out / "redesigned_proteins.fasta"))
            assert redesigned and all(r.idr_start == 2 for r in redesigned)
            assert all(r.full_seq[:2] == "AC" and r.full_seq.endswith("DE") for r in redesigned)
        if name.startswith("02"):
            assert np.load(out / "embeddings.npy").shape == (8, 16)
            assert namespace["residue_table"].protein_position_1based.tolist() == list(range(3, 19))
        if name.startswith(("03", "04")):
            monkeypatch.setattr(
                IDiomSAE, "from_pretrained", lambda *a, **k: pytest.fail("Reopened results loaded a model")
            )
            reuse = dict(parameters, OUT_DIR=tmp_path / "reopened")
            if name.startswith("03"):
                reuse["FEATURE_DIR"] = out / "features"
            else:
                reuse["RESULT_DIR"] = out
            execute_notebook(name, reuse)
        if name.startswith(("05", "06", "07")):
            restored = IDiom.from_pretrained(out / "model", device="cpu")
            assert any(
                not torch.equal(a, b) for a, b in zip(host.model.parameters(), restored.model.parameters())
            )
            assert (out / "training/checkpoints/last.ckpt").is_file()
        if name.startswith("07"):
            assert (out / "signature_coverage.csv").is_file()
            assert (out / "target_feature_presence.csv").is_file()
    finally:
        sae_feature._sae.cache_clear()
        sae_feature._target_ids.cache_clear()
        plt.close("all")
        torch.set_num_threads(previous_threads)


def test_split_keeps_duplicate_idrs_together():
    from idiom.data.records import Record
    from idiom.utils.notebook_helpers import idr_sequence, split_records

    records = [Record(str(i), s, 0, len(s)) for i, s in enumerate(["AAA", "CCC", "AAA", "DDD"])]
    train, validation = split_records(records, seed=2)
    assert train and validation
    assert set(map(idr_sequence, train)).isdisjoint(map(idr_sequence, validation))


def test_notebook_structure_and_links():
    """Keep the release set numbered, output-free, and independent of companion Python files."""
    import re

    assert sorted(p.stem for p in NOTEBOOKS.glob("*.ipynb")) == NOTEBOOK_NAMES
    for name in NOTEBOOK_NAMES:
        path = NOTEBOOKS / f"{name}.ipynb"
        notebook = json.loads(path.read_text())
        parameters = []
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            assert "workflow_utils" not in source
            if cell["cell_type"] == "code":
                compile(source, str(path), "exec")
                assert cell["execution_count"] is None and not cell["outputs"]
                parameters.extend(tag for tag in cell["metadata"].get("tags", []) if tag == "parameters")
            else:
                for dest in re.findall(r"\]\(([^)]+)\)", source):
                    if "github/rotskoff-group/idiom/blob/v1/" in dest:
                        assert dest.endswith(f"/cookbook/notebooks/{name}.ipynb")
                    elif "://" not in dest and not dest.startswith("#"):
                        assert (path.parent / dest.split("#")[0]).exists(), (name, dest)
        assert len(parameters) == 1
