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
    "generate_idrs",
    "extract_embeddings",
    "enriched_sae_features",
    "sft",
    "rl_custom",
    "rl_sae",
]


def execute_notebook(name, parameters, *, stop_after_tag=None):
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
        if stop_after_tag and stop_after_tag in cell["metadata"].get("tags", []):
            break
    return namespace


@pytest.mark.parametrize(
    "name,additional",
    [(n, None) for n in NOTEBOOK_NAMES] + [(NOTEBOOK_NAMES[-1], "second"), ("generate_idrs", "invalid")],
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
        N_PROTEINS=8,
        N=4,
        MAX_NEW_TOKENS=12,
        MAX_STEPS=2,
        TARGET_LENGTH=8,
        MAX_BACKGROUND=8,
        SIGNATURE_FILE=signature,
        SIGNATURE_NAME="demo",
        ADDITIONAL_SIGNATURE=additional,
        PROMPT_FASTA=positive,
        SAE_DEVICE="cpu",
    )
    import huggingface_hub

    from idiom.utils import notebook_helpers

    monkeypatch.setattr(notebook_helpers, "example_file", lambda *args: positive)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(background))
    if name in {"generate_idrs", "extract_embeddings"}:
        # Force an interior predicted span so the FASTA handoff must retain both flanks.
        from types import SimpleNamespace

        import metapredict

        monkeypatch.setattr(
            metapredict,
            "predict_disorder_batch",
            lambda sequences, **kwargs: [
                SimpleNamespace(
                    sequence=seq, disorder=np.ones(len(seq)), disordered_domain_boundaries=[[2, 18]]
                )
                for seq in sequences
            ],
        )
        unlabeled = tmp_path / "unlabeled.fasta"
        unlabeled.write_text(
            ">protein\nAX\n" if additional == "invalid" else ">protein\nAC" + "Q" * 16 + "DE\n"
        )
        parameters["INPUT_FASTA"] = unlabeled
    try:
        namespace = execute_notebook(name, parameters)
        out = parameters["OUT_DIR"]
        assert json.loads((out / "run.json").read_text())["elapsed_seconds"] > 0
        assert list(out.rglob("*.csv")) or (out / "enrichment.tsv").exists()
        if name == "generate_idrs":
            assert (out / "generated.fasta").exists()
            assert len(namespace["sequences"]) == parameters["N"]
            for source_name, generated_name in [
                ("prepared_records", "prepared_generated"),
                ("predicted_records", "predicted_generated"),
            ]:
                sources = {r.accession: r for r in namespace[source_name]}
                generated = namespace[generated_name]
                if sources:
                    assert generated
                for record in generated:
                    original = sources[record.accession.rsplit("_idiom_prompted_gen", 1)[0]]
                    assert record.idr_start == original.idr_start
                    assert record.full_seq[: record.idr_start] == original.full_seq[: original.idr_start]
                    assert record.full_seq[record.idr_end :] == original.full_seq[original.idr_end :]
            predicted = out / "predicted"
            annotated = list(read_records(predicted / "annotated_proteins.fasta"))
            idrs, _ = load_inputs(predicted / "idrs.fasta", "idr", None)
            assert [r.full_seq[r.idr_start : r.idr_end] for r in annotated] == [r.full_seq for r in idrs]
            assert (predicted / "prediction_settings.json").exists()
            if additional == "invalid":
                assert namespace["regions"].empty and not namespace["predicted_generated"]
                assert not (predicted / "redesigned_proteins.fasta").exists()
            else:
                assert annotated and namespace["predicted_generated"]
        if name == "extract_embeddings":
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
            assert namespace["sae_residues"].shape == (32, 32)
            assert namespace["sae_accessions"] == [r.accession for r in namespace["examples"]]
            np.testing.assert_allclose(
                namespace["sae_pooled"], namespace["sae_residues"].reshape(2, 16, 32).mean(1), atol=1e-6
            )
            assert [row["source_pos"] for row in namespace["sae_index"]] == list(range(16)) * 2
        if name == "enriched_sae_features":
            from idiom.sae.features import load_enrichment

            result, selection = load_enrichment(out / "enrichment.npz")
            assert result["n_pos"] == result["n_neg"] == 8
            assert len(selection["selected"]) == 32
        if name in ("sft", "rl_custom", "rl_sae"):
            restored = IDiom.from_pretrained(out / "model", device="cpu")
            assert any(
                not torch.equal(a, b) for a, b in zip(host.model.parameters(), restored.model.parameters())
            )
            assert (out / "training/checkpoints/last.ckpt").is_file()
        if name in ("rl_custom", "rl_sae"):
            import pandas as pd

            scores = pd.read_csv(out / "rewards.csv")
            assert set(scores.group) == {"baseline", "adapted"}
            assert np.isfinite(scores.total_reward).all()
        if name == "rl_sae":
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


def test_notebook_structure_and_links():
    """Keep notebooks free of saved errors, and independent of companion Python files."""
    import re

    from IPython.core.inputtransformer2 import TransformerManager

    assert sorted(p.stem for p in NOTEBOOKS.glob("*.ipynb")) == sorted(NOTEBOOK_NAMES)
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


def test_embedding_preparation_without_idrs(tmp_path, monkeypatch):
    """The fixed protein example stops before model loading when no regions are found."""
    from types import SimpleNamespace

    import metapredict

    from idiom.utils import notebook_helpers

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">protein\nACDEFGHIKLMNPQRSTVWY\n")
    monkeypatch.setattr(notebook_helpers, "example_file", lambda *args: fasta)
    monkeypatch.setattr(
        metapredict,
        "predict_disorder_batch",
        lambda sequences, **kwargs: [
            SimpleNamespace(sequence=s, disorder=np.zeros(len(s)), disordered_domain_boundaries=[])
            for s in sequences
        ],
    )
    monkeypatch.setattr(IDiom, "from_pretrained", lambda *args, **kwargs: pytest.fail("Loaded a model"))
    with pytest.raises(ValueError, match="No IDRs predicted"):
        execute_notebook("extract_embeddings", dict(OUT_DIR=tmp_path / "outputs", DEVICE="cpu"))
