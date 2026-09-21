"""Exercise the CLI with real enrichment statistics and synthetic SAE activations."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from idiom.sae.features import load_enrichment, select_features
from idiom.sae.features.enrichment_cli import main


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """Create enrichment FASTA inputs and substitute a deterministic SAE fixture."""
    positive = tmp_path / "positive.fasta"
    background = tmp_path / "background.fasta"
    positive.write_text("".join(f">p{i}\n{'A' * 15}\n" for i in range(30)))
    background.write_text(">overlap\n" + "A" * 15 + "\n" + "".join(f">b{i}\n{'C' * 15}\n" for i in range(60)))
    seen = []

    class FakeSAE:
        """Write deterministic feature datasets without loading a trained SAE."""

        fim_mode = "unprompted"
        region = "idr"
        host_model = "fake-host"
        layer = 1
        model = SimpleNamespace(cfg=SimpleNamespace(max_seq_len=100))
        sae = SimpleNamespace(num_latents=3)

        def build_feature_dataset(self, records, out, batch_size):
            """Write deterministic feature arrays and metadata while recording the input records."""
            seen.append(records)
            out.mkdir()
            feature = 0 if records[0].accession.startswith("p") else 1
            np.save(out / "top_indices.npy", np.full((len(records), 1), feature))
            np.save(out / "top_values.npy", np.ones((len(records), 1)))
            np.save(out / "seq_idx.npy", np.arange(len(records)))
            np.save(out / "pos_idx.npy", np.full(len(records), 10))
            (out / "strings.json").write_text(json.dumps(["132" + "A" * 15] * len(records)))
            (out / "meta.json").write_text(json.dumps({"num_latents": 3}))
            return out

    monkeypatch.setattr("idiom.IDiomSAE.from_pretrained", lambda *a, **kw: FakeSAE())
    return positive, background, seen


def test_cli_downloads_background_and_exports(tmp_path, monkeypatch, inputs):
    """Verify background download, overlap removal, and enrichment exports."""
    positive, background, seen = inputs
    downloads = []

    def download(*args, **kwargs):
        """Record download arguments and return the local background FASTA."""
        downloads.append((args, kwargs))
        return str(background)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    out = tmp_path / "out"
    main(["--sae", "fake", "--positive", str(positive), "--out", str(out), "--name", "demo"])
    assert downloads == [
        (("jxliu2/idiom-db", "idiom-db/idiom-db-v1_validation.fasta"), {"repo_type": "dataset"})
    ]
    assert len(seen[1]) == 60
    assert all(r.accession != "overlap" for r in seen[1])
    assert json.loads((out / "signature.json").read_text())["top30"]["demo"] == [0]
    run = json.loads((out / "run.json").read_text())
    assert (run["n_pos"], run["n_background"]) == (30, 60)
    with (out / "enrichment.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 3 and rows[0]["selected"] == "True"
    assert rows[2]["active"] == "False"
    result, saved = load_enrichment(out / "enrichment.npz")
    recomputed = select_features(result, feature_dir=out / "fd_background")
    assert saved["ids"] == recomputed["ids"] == [0]
    for key in ("enriched", "boundary_filtered", "selected"):
        np.testing.assert_array_equal(saved[key], recomputed[key])


def test_cli_local_background_no_signature_and_no_stale_rerun(tmp_path, monkeypatch, inputs):
    """Verify local-background use, optional signatures, and rejection of stale output reuse."""
    positive, background, _ = inputs
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **kw: pytest.fail("unexpected download")
    )
    out = tmp_path / "out"
    args = [
        "--sae",
        "fake",
        "--positive",
        str(positive),
        "--background",
        str(background),
        "--out",
        str(out),
        "--name",
        "demo",
        "--log2or-floor",
        "100",
        "--max-positive",
        "10",
    ]
    main(args)
    assert not (out / "signature.json").exists()
    assert json.loads((out / "run.json").read_text())["n_pos"] == 10
    with pytest.raises(SystemExit):
        main(args)


def test_cli_rejects_empty_background(tmp_path, inputs):
    """Verify that an empty background is rejected before feature extraction."""
    positive, _, seen = inputs
    with pytest.raises(SystemExit):
        main(
            [
                "--sae",
                "fake",
                "--positive",
                str(positive),
                "--background",
                str(positive),
                "--out",
                str(tmp_path / "out"),
                "--name",
                "demo",
            ]
        )
    assert not seen
