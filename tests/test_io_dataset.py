"""P1 tests (CPU-only): FASTA reader + drop policy, header parsing, record dataset."""

import torch

from idiom.data.dataset import RecordDataset, make_collate, max_protein_len, record_to_example
from idiom.data.io import Record, parse_idr_header, read_fasta, read_records
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
    # P00002 has an 'X' -> dropped; the other two survive.
    accs = [h.split("_IDR_")[0] for h, _ in pairs]
    assert accs == ["P00001", "P00003"]


def test_parse_idr_header():
    # 1-indexed inclusive in the header -> 0-indexed half-open internally.
    assert parse_idr_header("P06748_IDR_119-242") == ("P06748", 118, 242)
    assert parse_idr_header("P00001_IDR_3-6 trailing") == ("P00001", 2, 6)


def test_read_records(tmp_path):
    recs = list(read_records(_write(tmp_path, FASTA)))
    assert [r.accession for r in recs] == ["P00001", "P00003"]
    r = recs[0]
    assert (r.full_seq, r.idr_start, r.idr_end) == ("MEDSKVDNRPQ", 2, 6)


def test_record_to_example_shift():
    rec = Record("P0", "MEDSKVDNRPQ", 4, 8)  # IDR = seq[4:8] = "KVDN" (half-open)
    x, y = record_to_example(rec, TOK, variant="full")
    assert x.shape == y.shape
    assert x[0].item() == TOK.start_id and y[-1].item() == TOK.stop_id
    # the shift: input[1:] == target[:-1], and it decodes to the FIM string.
    assert torch.equal(x[1:], y[:-1])
    assert TOK.decode(x[1:].tolist()) == "1MEDS3RPQ2KVDN"


def test_dataset_len_filter_and_getitem():
    keep = max_protein_len(16)  # 16 - 4 = 12
    recs = [
        Record("ok", "MEDSKVDNRPQ", 2, 5),   # len 11 <= 12 -> kept
        Record("too_long", "A" * 20, 0, 19),  # len 20 > 12 -> dropped
    ]
    ds = RecordDataset(recs, TOK, max_len=16, fim_full_prob=1.0)
    assert len(ds) == 1 and keep == 12
    x, y, m = ds[0]
    assert x[0].item() == TOK.start_id and x.dtype == torch.long
    assert m.shape == y.shape and m.all()  # pretraining default: loss on all tokens


def test_collate_pads():
    recs = [Record("a", "MEDSKVDNRPQ", 2, 5), Record("b", "ACDEFGHIKL", 1, 8)]
    ds = RecordDataset(recs, TOK, max_len=64, fim_full_prob=1.0)
    collate = make_collate(TOK.pad_id)
    x, y, m = collate([ds[0], ds[1]])
    assert x.shape == y.shape == m.shape and x.size(0) == 2
    # shorter row is padded out to the batch max length.
    assert (x == TOK.pad_id).any()
