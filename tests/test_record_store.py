"""Record store (memory-mapped columnar dataset backing) — parity with read_records + the dataset."""

from __future__ import annotations

import torch

from idiom.data.dataset import RecordDataset
from idiom.data.io import read_records
from idiom.data.record_store import RecordStore, build_record_store, open_or_build, store_path_for

# Two valid records, one non-canonical (X -> dropped, D15), one malformed header (no _IDR_),
# one out-of-range span (end > len -> dropped). read_records keeps exactly the first two + the
# 5th (valid). Sequences are multi-line-wrapped to exercise the parser.
FASTA = """\
>P1_IDR_2-5
MKLVQRST
>P2_IDR_1-3
ACD
EFG
>P3_IDR_1-3
ACXDEF
>BADHEADER
ACDEFG
>P5_IDR_3-9
MNPQRSTVWY
"""


def _write(tmp_path):
    p = tmp_path / "recs.fasta"
    p.write_text(FASTA)
    return p


def test_store_matches_read_records(tmp_path):
    fasta = _write(tmp_path)
    expected = list(read_records(fasta))
    store = RecordStore(build_record_store(fasta))
    assert len(store) == len(expected) == 3
    for i, rec in enumerate(expected):
        got = store[i]
        assert (got.accession, got.full_seq, got.idr_start, got.idr_end) == (
            rec.accession, rec.full_seq, rec.idr_start, rec.idr_end
        )


def test_seq_lengths_vectorized(tmp_path):
    fasta = _write(tmp_path)
    store = RecordStore(build_record_store(fasta))
    assert store.seq_lengths().tolist() == [len(r.full_seq) for r in read_records(fasta)]


def test_dataset_parity_store_vs_list(tmp_path):
    """Store-backed and list-backed RecordDataset yield identical (x, y, mask) at every index."""
    fasta = _write(tmp_path)
    for completion_only in (False, True):
        ds_list = RecordDataset(read_records(fasta), seed=7, completion_only=completion_only)
        ds_store = RecordDataset(open_or_build(fasta), seed=7, completion_only=completion_only)
        assert len(ds_list) == len(ds_store)
        for i in range(len(ds_list)):  # same seed + same record order => same per-sample rng draw
            xl, yl, ml = ds_list[i]
            xs, ys, ms = ds_store[i]
            assert torch.equal(xl, xs) and torch.equal(yl, ys) and torch.equal(ml, ms)


def test_length_filter_parity(tmp_path):
    fasta = _write(tmp_path)
    # max_len small enough that the 10-residue P5 record (full example = 14 positions) is dropped.
    max_len = 12
    ds_list = RecordDataset(read_records(fasta), max_len=max_len)
    ds_store = RecordDataset(open_or_build(fasta), max_len=max_len)
    assert len(ds_store) == len(ds_list) < 3


def test_open_or_build_caches_and_rebuilds_on_change(tmp_path):
    fasta = _write(tmp_path)
    store_dir = store_path_for(fasta)
    s1 = open_or_build(fasta)
    assert store_dir.is_dir() and len(s1) == 3
    mtime1 = (store_dir / "meta.json").stat().st_mtime_ns
    open_or_build(fasta)  # valid cache -> must not rebuild
    assert (store_dir / "meta.json").stat().st_mtime_ns == mtime1

    fasta.write_text(FASTA + ">P6_IDR_1-4\nKLMNQRST\n")  # source changed -> stale -> rebuild
    s2 = open_or_build(fasta)
    assert len(s2) == 4
