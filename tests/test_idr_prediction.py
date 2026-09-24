"""Check predicted region coordinates and downstream FASTA handoffs."""

from types import SimpleNamespace

import numpy as np

from idiom.data.records import read_records
from idiom.utils.idr_prediction import predict_idrs_fasta
from idiom.utils.notebook_helpers import load_inputs


def test_prediction_exports_and_audit(tmp_path, monkeypatch):
    import metapredict

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">same\nACDEFGHIKL\n>same\nACDEFGHIKL\n>folded\nAAAA\n>bad\nAX\n>\nACDE\n")

    def predict(sequences, **kwargs):
        assert kwargs["version"] == "V3" and kwargs["return_domains"]
        return [
            SimpleNamespace(
                sequence=s,
                disorder=np.ones(len(s)),
                disordered_domain_boundaries=[[1, 4], [6, 10]] if len(s) == 10 else [],
            )
            for s in sequences
        ]

    monkeypatch.setattr(metapredict, "predict_disorder_batch", predict)
    out = tmp_path / "out"
    regions, audit, scores = predict_idrs_fasta(fasta, out, minimum_idr_length=3)
    assert regions.start_1based.tolist() == [2, 7, 2, 7]
    assert regions.end_1based.tolist() == [4, 10, 4, 10]
    assert regions.region_id.nunique() == 4
    assert audit.status.tolist()[:3] == ["predicted IDRs", "predicted IDRs", "no predicted IDRs"]
    assert len(scores) == 3
    annotated = list(read_records(out / "annotated_proteins.fasta"))
    isolated, _ = load_inputs(out / "idrs.fasta", "idr", None)
    assert [r.full_seq for r in isolated] == ["CDE", "HIKL"] * 2
    assert [r.full_seq[r.idr_start : r.idr_end] for r in annotated] == [r.full_seq for r in isolated]
    assert len(load_inputs(out / "annotated_proteins.fasta", "annotated", None)[0]) == 4


def test_empty_prediction_exports(tmp_path):
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")
    regions, audit, scores = predict_idrs_fasta(fasta, tmp_path / "out")
    assert regions.empty and audit.empty and not scores
    assert (tmp_path / "out/idrs.fasta").read_text() == ""
