# Legacy reference scripts (v1) — read-only

Verbatim copies from `idr-plm-figures/src/idr_plm_figures/data_preprocess` (the most up-to-date
v1 repo). **Reference only** — kept here so the v2 ports can match the known-good logic. Not run.
The v1 path materialized FIM into HDF5; v2 keeps the store as raw `full_seq` + coords and does FIM
on the fly, so these get *ported* (filters extracted), not reused as-is.

## What each does (and the two filter steps requested for v2)

| file | stage | role |
|---|---|---|
| `AFDB_90_mmseqs2.bash` | cluster | mmseqs linclust 90% → `reps.fasta` (53M reps) |
| `find_fastas_in_h5.py` / `.bash` | build master | pull every IDR whose base accession is a cluster rep → **`AFDB_IDR_90_alldata.h5`** (73M, the curated master). Pure accession match; **no** length/pLDDT filter. |
| `make_AFDB_FIM.py` / `.bash` | **filter + FIM** | reads the 64 parts + a target FASTA, applies the two filters below, FIM-transforms, writes `AFDB_IDR_90_FIM_512.h5` (≈37M training set). |

## The two v2 processing steps live in `make_AFDB_FIM.py`

1. **Length filter** — `MAX_LEN = 512`; `if len_arr[idx] > MAX_LEN: continue`.
   → v2: cap is `full_seq ≤ max_len − 4` (room for 3 FIM markers + START), i.e. **≤ 1020** for a
   1024-token model. (Not literally 1024.)

2. **Fully-low-pLDDT removal** — `if start_pos == 0 and end_pos == len(seq_full) - 1: continue`.
   A whole-protein IDR (`[0, len-1]`) means the extractor found **no folded segment**, i.e. the
   protein has no region above the folded threshold → "fully low-pLDDT". This skip is exactly the
   "~1/3 fully low-pLDDT" removal described in the paper SI (and visualized by SI Fig S2
   `corpus_plddt`). **v2 standardizes on this aggressive form**: `filter_length_plddt.is_fully_low_plddt`
   (= `not extract.has_folded_segment`) drops any protein with no surviving folded *segment* —
   stricter than `max(full_avg_plddt) < 80`, since it also removes proteins whose only high-pLDDT
   residues form a sub-`min_seg_length` blip (which a max-threshold would keep).

`make_AFDB_FIM.py` also emits **two FIM variants per IDR** (full `1pre3suf2mid` + middle-only
`132mid`) and skips the degenerate full-span case — in v2 those variants are produced on the fly
by `idiom.data.fim`, so only the *filter* logic carries over.
