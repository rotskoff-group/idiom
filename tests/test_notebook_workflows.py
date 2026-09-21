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
    "02_predict_idrs",
    "03_extract_embeddings",
    "04_interpret_sae_features",
    "05_discover_feature_signature",
    "06_finetune_and_generate",
    "07_design_with_custom_rewards",
    "08_design_with_rl_sae",
]


def execute_notebook(name, parameters):
    """Execute cells against the local install, skipping installation and replacing settings."""
    from IPython.core.inputtransformer2 import TransformerManager

    transformer = TransformerManager()
    namespace = {"__name__": "__main__"}
    notebook = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text())
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        # Exercise this checkout without installing the published release over it.
        if "installation" in cell["metadata"].get("tags", []):
            continue
        source = "".join(cell["source"])
        exec(compile(transformer.transform_cell(source), f"{name}:cell_{i}", "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(parameters)
    return namespace


@pytest.mark.parametrize(
    "name,additional", [(n, None) for n in NOTEBOOK_NAMES] + [(NOTEBOOK_NAMES[-1], "second")]
)
def test_notebook_execution(name, additional, tmp_path, monkeypatch):
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
        MAX_STEPS=2,
        TARGET_LENGTH=8,
        POSITIVE_FASTA=positive,
        POSITIVE_MODE="annotated",
        BACKGROUND_FASTA=background,
        BACKGROUND_MODE="annotated",
        MAX_POSITIVE=8,
        MAX_BACKGROUND=8,
        SIGNATURE_FILE=signature,
        SIGNATURE_NAME="demo",
        ADDITIONAL_SIGNATURE=additional,
        PROMPT_FASTA=positive,
        SAE_DEVICE="cpu",
    )
    try:
        namespace = execute_notebook(name, parameters)
        out = parameters["OUT_DIR"]
        assert json.loads((out / "run.json").read_text())["elapsed_seconds"] > 0
        assert list(out.glob("*.csv")) or (out / "enrichment.tsv").exists()
        if name.startswith("01"):
            redesigned = list(read_records(out / "redesigned_proteins.fasta"))
            assert redesigned and all(r.idr_start == 2 for r in redesigned)
            assert all(r.full_seq.startswith("AC") and r.full_seq.endswith("DE") for r in redesigned)
            assert (out / "generated.fasta").exists()
            assert len(namespace["sequences"]) == parameters["N"]
        if name.startswith("02"):
            assert (out / "prediction_settings.json").exists()
            annotated = list(read_records(out / "annotated_proteins.fasta"))
            isolated_records, _ = load_inputs(out / "idrs.fasta", "idr", None)
            assert [r.full_seq[r.idr_start : r.idr_end] for r in annotated] == [
                r.full_seq for r in isolated_records
            ]
            import pandas as pd

            candidates = pd.read_csv(out / "prompted_sequences.csv", keep_default_na=False)
            prompts = {r.accession: r for r in annotated[: namespace["MAX_PROMPT_REGIONS"]]}
            assert len(candidates) == len(prompts) * namespace["N_PER_REGION"]
            mapping = candidates.set_index("generated_id")
            regenerated = list(read_records(out / "redesigned_proteins.fasta"))
            assert len(regenerated) == (candidates.status == "generated").sum()
            for record in regenerated:
                row = mapping.loc[record.accession]
                original = prompts[row.region_id]
                assert record.idr_start == original.idr_start
                assert record.full_seq[: record.idr_start] == original.full_seq[: original.idr_start]
                assert record.full_seq[record.idr_end :] == original.full_seq[original.idr_end :]
                assert record.full_seq[record.idr_start : record.idr_end] == row.generated_idr
                assert row.original_idr == original.full_seq[original.idr_start : original.idr_end]
        if name.startswith("03"):
            assert np.load(out / "embeddings.npy").shape == (8, 16)
            import pandas as pd

            index = pd.read_csv(out / "embedding_index.csv")
            assert index.embedding_row.tolist() == list(range(8))
            assert index.sequence.str.len().tolist() == [16] * 8
            residue = np.load(out / "residue_embeddings.npy")
            last = np.load(out / "last_embeddings.npy")
            assert residue.shape == (32, 16) and last.shape == (2, 16)
            np.testing.assert_allclose(last, residue[[15, 31]], atol=1e-6)
            np.testing.assert_allclose(
                np.load(out / "embeddings.npy")[:2], residue.reshape(2, 16, 16).mean(1), atol=1e-6
            )
            positions = pd.read_csv(out / "residue_index.csv")
            assert positions.protein_position_1based.tolist() == list(range(3, 19)) * 2
        if name.startswith("04"):
            assert (out / "features/meta.json").exists()
            assert list(out.glob("*_trace.png")) and list(out.glob("*_logo.png"))
        if name.startswith("05"):
            from idiom.sae.features import load_enrichment

            result, selection = load_enrichment(out / "enrichment.npz")
            assert result["n_pos"] == result["n_neg"] == 8
            assert len(selection["selected"]) == 32
        if name.startswith(("06", "07", "08")):
            restored = IDiom.from_pretrained(out / "model", device="cpu")
            assert any(
                not torch.equal(a, b) for a, b in zip(host.model.parameters(), restored.model.parameters())
            )
            assert (out / "training/checkpoints/last.ckpt").is_file()
        if name.startswith(("07", "08")):
            import pandas as pd

            scores = pd.read_csv(out / "rewards.csv")
            assert set(scores.group) == {"baseline", "adapted"}
            assert np.isfinite(scores.total_reward).all()
        if name.startswith("08"):
            expected = {"demo": [0, 1]}
            if additional:
                expected.update(second=[1, 2], combined=[0, 1, 2])
                coverage = pd.read_csv(out / "signature_coverage.csv")
                assert set(coverage.signature) == set(expected)
                assert coverage.coverage.between(0, 1).all()
            assert json.loads((out / "signature.json").read_text())["top30"] == expected
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


def test_prediction_notebook_without_valid_proteins(tmp_path, monkeypatch):
    """An empty prediction set exports valid empty tables without loading a generator."""
    import matplotlib

    matplotlib.use("Agg")
    fasta = tmp_path / "invalid.fasta"
    fasta.write_text(">invalid\nAX\n")
    monkeypatch.setattr(IDiom, "from_pretrained", lambda *a, **k: pytest.fail("Loaded a generator"))
    out = tmp_path / "outputs"
    namespace = execute_notebook("02_predict_idrs", {"INPUT_FASTA": fasta, "OUT_DIR": out, "DEVICE": "cpu"})
    assert namespace["regions"].empty and namespace["candidates"].empty
    assert (out / "prompted_idrs.fasta").read_text() == ""
    assert (out / "redesigned_proteins.fasta").read_text() == ""
    assert (out / "prompted_sequences.csv").read_text().startswith("generated_id,region_id,")


def test_notebook_structure_and_links():
    """Keep notebooks numbered, free of saved errors, and independent of companion Python files."""
    import re

    from IPython.core.inputtransformer2 import TransformerManager

    assert sorted(p.stem for p in NOTEBOOKS.glob("*.ipynb")) == NOTEBOOK_NAMES
    for name in NOTEBOOK_NAMES:
        path = NOTEBOOKS / f"{name}.ipynb"
        notebook = json.loads(path.read_text())
        parameters = []
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            assert "workflow_utils" not in source
            if cell["cell_type"] == "code":
                compile(TransformerManager().transform_cell(source), str(path), "exec")
                assert all(output["output_type"] != "error" for output in cell["outputs"])
                parameters.extend(tag for tag in cell["metadata"].get("tags", []) if tag == "parameters")
            else:
                for dest in re.findall(r"\]\(([^)]+)\)", source):
                    if "github/rotskoff-group/idiom/blob/v1/" in dest:
                        target = dest.rsplit("/", 1)[-1]
                        assert target.removesuffix(".ipynb") in NOTEBOOK_NAMES
                    elif "://" not in dest and not dest.startswith("#"):
                        assert (path.parent / dest.split("#")[0]).exists(), (name, dest)
        assert len(parameters) == 1
