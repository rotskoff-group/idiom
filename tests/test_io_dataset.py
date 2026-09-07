"""Tests: FASTA reader + drop policy, header parsing, record dataset."""

import torch

from idiom.data.dataset import RecordDataset, make_collate, max_protein_len, record_to_example
from idiom.data.io import Record, parse_idr_header, read_fasta, read_records, to_records
from idiom.data.tokenizer import Tokenizer

TOK = Tokenizer()

FASTA = """\
>P00001_IDR_3-6
MEDSKVDNRPQ
>P00002_IDR_1-4 some description
MKLXVGQ
>P00003_IDR_2-3
ACDEFG
"""


def _write(tmp_path, text):
    p = tmp_path / "records.fasta"
    p.write_text(text)
    return p


def test_read_fasta_drops_noncanonical(tmp_path):
    pairs = read_fasta(_write(tmp_path, FASTA))
    accs = [h.split("_IDR_")[0] for h, _ in pairs]
    assert accs == ["P00001", "P00003"]


def test_parse_idr_header():
    assert parse_idr_header("P06748_IDR_119-242") == ("P06748", 118, 242)
    assert parse_idr_header("P00001_IDR_3-6 trailing") == ("P00001", 2, 6)


def test_read_records(tmp_path):
    recs = list(read_records(_write(tmp_path, FASTA)))
    assert [r.accession for r in recs] == ["P00001", "P00003"]
    r = recs[0]
    assert (r.full_seq, r.idr_start, r.idr_end) == ("MEDSKVDNRPQ", 2, 6)


def test_record_to_example_shift():
    rec = Record("P0", "MEDSKVDNRPQ", 4, 8)  # IDR = seq[4:8] = "KVDN" (half-open)
    x, y = record_to_example(rec, TOK, variant="prompted")
    assert x.shape == y.shape
    assert x[0].item() == TOK.start_id and y[-1].item() == TOK.stop_id
    assert torch.equal(x[1:], y[:-1])
    assert TOK.decode(x[1:].tolist()) == "1MEDS3RPQ2KVDN"


def test_dataset_len_filter_and_getitem():
    keep = max_protein_len(16)
    recs = [
        Record("ok", "MEDSKVDNRPQ", 2, 5),
        Record("too_long", "A" * 20, 0, 19),
    ]
    ds = RecordDataset(recs, TOK, max_len=16, prompted_prob=1.0)
    assert len(ds) == 1 and keep == 12
    x, y, m = ds[0]
    assert x[0].item() == TOK.start_id and x.dtype == torch.long
    assert m.shape == y.shape and m.all()


def test_collate_pads():
    recs = [Record("a", "MEDSKVDNRPQ", 2, 5), Record("b", "ACDEFGHIKL", 1, 8)]
    ds = RecordDataset(recs, TOK, max_len=64, prompted_prob=1.0)
    collate = make_collate(TOK.pad_id)
    x, y, m = collate([ds[0], ds[1]])
    assert x.shape == y.shape == m.shape and x.size(0) == 2
    assert (x == TOK.pad_id).any()


def test_to_records_normalizes_inputs(tmp_path):
    recs = list(to_records("MEDSKVDN"))
    assert len(recs) == 1
    assert (recs[0].accession, recs[0].idr_start, recs[0].idr_end) == ("seq_0", 0, 8)
    recs = list(to_records(["MEDS", "ACDE"]))
    assert [r.accession for r in recs] == ["seq_0", "seq_1"]
    r = Record("X", "MEDS", 1, 3)
    assert list(to_records(r)) == [r] and list(to_records([r])) == [r]
    fa = tmp_path / "p.fasta"
    fa.write_text(">A_IDR_2-4\nMEDSKV\n")
    recs = list(to_records(fa))
    assert len(recs) == 1 and recs[0].accession == "A"


def test_to_records_noncanonical_sequence_raises():
    import pytest

    with pytest.raises(ValueError, match="canonical"):
        list(to_records("MEDSX"))


def test_long_bare_sequence_matches_list_input():
    seq = "ACDEFGHIKLMNPQRSTVWY" * 15
    assert list(to_records(seq)) == list(to_records([seq]))


def test_existing_sequence_named_file_keeps_path_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ACDE").write_text(">protein_IDR_1-4\nMKLV\n")
    assert list(to_records("ACDE")) == [Record("protein", "MKLV", 0, 4)]
