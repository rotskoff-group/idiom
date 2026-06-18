"""P1 curation test (CPU-only): DisProt dedup helpers. No mmseqs / pipeline run."""

import json

from extras.data_pipeline.dedup import disprot_idr_fasta, remove_headers, write_idr_fasta


def _read_fasta(path):
    out, header = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            header = line[1:]
        elif header is not None:
            out.append((header, line))
            header = None
    return out


def test_disprot_idr_fasta_extracts_D_regions(tmp_path):
    seq = "A" * 100
    data = [
        {
            "acc": "P1",
            "sequence": seq,
            "disprot_consensus": {"Structural state": [
                {"type": "D", "start": 1, "end": 40},   # 40-residue IDR -> kept
                {"type": "S", "start": 41, "end": 60},   # structured -> skipped
                {"type": "D", "start": 61, "end": 75},   # 15-residue IDR -> below min_idr_length
            ]},
        }
    ]
    jp = tmp_path / "dp.json"
    jp.write_text(json.dumps(data))
    out = tmp_path / "dp_idrs.fasta"
    n = disprot_idr_fasta(str(jp), str(out), min_idr_length=30)
    recs = _read_fasta(out)
    assert n == 1 and len(recs) == 1
    assert recs[0][1] == seq[0:40]  # 1-based inclusive [1,40] -> seq[0:40]


def test_idr_fasta_header_matches_record_and_removal(tmp_path):
    # a record FASTA: full_seq + _IDR_x-y; the IDR is a slice of it.
    full = "MKL" + "S" * 40 + "GG"  # IDR at 1-based 4..43 -> 0-based [3:43]
    rec = tmp_path / "records.fasta"
    rec.write_text(f">P1-F1_IDR_4-43\n{full}\n>P2-F1_IDR_4-43\n{full}\n")

    idrf = tmp_path / "idrs.fasta"
    write_idr_fasta(str(rec), str(idrf))
    recs = _read_fasta(idrf)
    # header token is preserved exactly, sequence is the IDR substring
    assert recs[0][0] == "P1-F1_IDR_4-43"
    assert recs[0][1] == full[3:43]

    # removing P1's header drops exactly that record
    out = tmp_path / "kept.fasta"
    kept, removed = remove_headers(str(rec), {"P1-F1_IDR_4-43"}, str(out))
    assert (kept, removed) == (1, 1)
    assert [h for h, _ in _read_fasta(out)] == ["P2-F1_IDR_4-43"]
