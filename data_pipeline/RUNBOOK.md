# Curation runbook (offline)

Produces the training **record store** — per-split `train/val/test.fasta` in the `_IDR_x-y`
convention (D16), capped at the model's protein-length budget. Heavy + operator-run; **not**
part of training or CI. The pure cores (`extract`, `filter`, `split`, `dedup`) are CPU-tested.
Ported/condensed from `idr-plm-figures/.../data_preprocess` (verbatim v1 refs in `_legacy_reference/`).

**v2 flow:** extract → cluster(90%) → master h5 → **filter** (length + fully-low-pLDDT) →
**DisProt dedup** → *[TM/SignalP/coiled-coil filter — future]* → **random split**. No FIM/precompute
materialization; the record FASTAs hold raw `full_seq` + coords and FIM happens on the fly.

**Length cap:** `full_seq ≤ max_len − 4` (3 FIM markers + START/STOP boundary). For a 1024-token
model that is **≤ 1020** (replaces the legacy `full_length ≤ 512`).

---

## 1. Extract IDRs (AFDB → records)  *(done — this is the existing master)*

For each AFDB part (`accession_ids`, `sequences`, `confidence_scores`), run the segmentation:

```python
from data_pipeline.extract import extract_idrs   # 0-based half-open spans
for acc, seq, plddt in afdb_part:
    for start, end in extract_idrs(plddt):        # thresholds: folded>80, dis<70, IDR 30–4096
        emit(acc, seq, start, end)                # one record per IDR
```

## 2. Cluster (mmseqs linclust, 90% id / 80% cov)  *(done)*

```bash
mmseqs createdb  alldata_seqs.fasta  DB
mmseqs linclust  DB  CLU  tmp  --min-seq-id 0.9 --cov-mode 0 -c 0.8 --cluster-mode 2
mmseqs createsubdb CLU DB REPS && mmseqs convert2fasta REPS reps.fasta
```

`find_fastas_in_h5.py` (`_legacy_reference/`) then pulls every IDR whose base accession is a rep
→ the curated master `clustering_90/AFDB_IDR_90_alldata.h5` (73M). No length/pLDDT filter here.

## 3. Filter → record FASTA  (`filter_length_plddt.py`)

The filter driver streams the 73M master, applies **both** filters, and writes the record FASTA
(`>{base}_IDR_{x}-{y}`, 1-based inclusive + `full_seq`):

```bash
python -m data_pipeline.filter_length_plddt --h5 .../AFDB_IDR_90_alldata.h5 --out train_candidates.fasta --max-len 1024
```

- **Length** — `passes_length`: keep `full_seq` length `≤ max_len − 4` → **≤ 1020**.
- **Fully-low-pLDDT** — `is_fully_low_plddt` (= `not extract.has_folded_segment`): drop proteins
  with **no folded segment** (the *aggressive* criterion — stricter than `max(full_avg_plddt) < 80`,
  also dropping sub-`min_seg_length` folded blips). Removes the "~1/3 fully low-pLDDT" set (SI;
  SI Fig S2 `corpus_plddt`). `full_avg_plddt` is already window-15 smoothed (check runs `window=1`).

## 4. DisProt leakage dedup  (`dedup.bash` / `dedup.py`)

Drop record IDRs ≥50% identical to any DisProt IDR (keeps DisProt a clean benchmark). **IDR-vs-IDR**
(a short DisProt IDR vs a full protein never meets 80% coverage), so we search the record FASTA's
IDR substrings and remove matched records from the full-seq FASTA:

```bash
srun -c 64 --mem 64GB -t 12:00:00 bash data_pipeline/dedup.bash train_candidates.fasta WORK_DIR
# -> WORK_DIR/train_dedup.fasta
```

(ESM-2 params: `mmseqs search --min-seq-id 0.5 -c 0.8 --cov-mode 0 -s 7 --max-seqs 300`.)

The DisProt query set comes from the **canonical parser** `data_pipeline/disprot.py` (ported from
v1 `idr-plm-figures` `utils.utils`), shared with the SI figures so dedup and figures dedup/compare
against the *exact same* set: 'D' consensus regions, IDR ≥ 30, full seq ≤ **1020** (v1 was 512;
raised to match the corpus cap), full IDPs removed by v1's fuzzy ±1 rule → **1,665** DisProt IDRs.

## 4b. TM / SignalP / coiled-coil filter  *(FUTURE — placeholder)*

Drop records whose protein has a transmembrane region (DeepTMHMM), signal peptide (SignalP), or
coiled-coil — these are ordered/structured features that pollute the IDR corpus. Slots in here,
**after** DisProt dedup and **before** split. Not yet implemented.

## 5. Random split (99 / 0.5 / 0.5)  (`split.py`)

Plain random partition over IDR **records** (per user 2026-06-14; not protein-grouped):

```bash
python -m data_pipeline.split --fasta train_dedup.fasta --out-dir splits/ --fractions 0.99 0.005 0.005
```

Writes `splits/{train,val,test}.fasta`, consumed directly by
`idiom.data.datamodule.RecordDataModule(train_fasta=..., val_fasta=..., test_fasta=...)`.
`read_records` re-checks length + canonical residues on load (belt-and-suspenders).
