"""P1 curation test (CPU-only): record filters (length + fully-low-pLDDT). No pipeline run."""

import numpy as np

from extras.data_pipeline.extract import has_folded_segment
from extras.data_pipeline.filter_length_plddt import is_fully_low_plddt, passes_length


def test_length_cap_is_max_len_minus_4():
    assert passes_length(1020, max_len=1024)
    assert not passes_length(1021, max_len=1024)
    assert not passes_length(1024, max_len=1024)


def test_fully_disordered_is_dropped():
    # all-low pLDDT -> no folded region -> fully disordered.
    assert is_fully_low_plddt(np.full(200, 30.0))


def test_real_folded_domain_is_kept():
    # a genuine folded domain (>= min_seg_length above threshold) + a disordered tail.
    plddt = np.concatenate([np.full(40, 90.0), np.full(160, 30.0)])
    assert has_folded_segment(plddt, window=1)
    assert not is_fully_low_plddt(plddt)


def test_aggressive_drops_subsegment_blip():
    # a 5-residue high-pLDDT blip (< min_seg_length) in a sea of disorder:
    # max(plddt) = 90 >= 80 (a max-threshold would KEEP it), but no folded *run* survives.
    plddt = np.concatenate([np.full(50, 30.0), np.full(5, 90.0), np.full(50, 30.0)])
    assert plddt.max() >= 80.0
    assert is_fully_low_plddt(plddt)  # aggressive criterion drops it


def _write_master_h5(path, rows):
    """rows: list of (acc_id, full_seq, idr_start, idr_end_incl, plddt_array)."""
    import h5py

    str_dt = h5py.string_dtype("utf-8")
    vlen = h5py.vlen_dtype(np.float16)
    with h5py.File(path, "w") as f:
        f.create_dataset("accession_ids", data=[r[0] for r in rows], dtype=str_dt)
        f.create_dataset("full_seq", data=[r[1] for r in rows], dtype=str_dt)
        f.create_dataset("full_length", data=[len(r[1]) for r in rows], dtype="i2")
        f.create_dataset("idr_start", data=[r[2] for r in rows], dtype="i2")
        f.create_dataset("idr_end", data=[r[3] for r in rows], dtype="i2")
        fap = f.create_dataset("full_avg_plddt", shape=(len(rows),), dtype=vlen)
        for i, r in enumerate(rows):
            fap[i] = r[4].astype(np.float16)


def test_write_filtered_fasta_applies_both_filters(tmp_path):
    from extras.data_pipeline.filter_length_plddt import write_filtered_fasta

    folded = np.concatenate([np.full(20, 90.0), np.full(30, 30.0)])  # 50aa, IDR 20..49
    keep = ("P1-F1_0-49", "M" * 50, 20, 49, folded)            # kept (has folded domain)
    low = ("P2-F1_0-29", "A" * 30, 0, 29, np.full(30, 30.0))  # dropped (fully low-pLDDT)
    long_ = ("P3-F1_0-49", "C" * 1100, 0, 49, np.concatenate([np.full(40, 90.0), np.full(1060, 30.0)]))  # dropped (>1020)
    h5 = tmp_path / "master.h5"
    _write_master_h5(str(h5), [keep, low, long_])

    out = tmp_path / "records.fasta"
    kept, total = write_filtered_fasta(str(h5), str(out), max_len=1024)
    assert (kept, total) == (1, 3)
    text = out.read_text().splitlines()
    assert text[0] == ">P1-F1_IDR_21-50"  # 0-based [20,49] inclusive -> 1-based [21,50]
    assert text[1] == "M" * 50
