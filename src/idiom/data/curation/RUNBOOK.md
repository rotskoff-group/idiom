# Curation runbook (offline)

Produces the training **record store** — per-split `train/val/test.fasta` in the `_IDR_x-y`
convention (D16), capped at the model's protein-length budget. Heavy + operator-run; **not**
part of training or CI. The pure segmentation core (`extract.extract_idrs`) is the only piece
under test here. Ported/condensed from `idr-plm-figures/.../data_preprocess`.

**Length cap:** `full_seq ≤ max_len − 4` (3 FIM markers + START). For a 1024-token model that
is **≤ 1020** (replaces the legacy `full_length ≤ 512`).

---

## 1. Extract IDRs (AFDB → records)

For each AFDB part (`accession_ids`, `sequences`, `confidence_scores`), run the segmentation:

```python
from idiom.data.curation.extract import extract_idrs   # 0-based half-open spans
for acc, seq, plddt in afdb_part:
    for start, end in extract_idrs(plddt):              # thresholds: folded>80, dis<70, IDR 30–4096
        emit(acc, seq, start, end)                      # one record per IDR
```

Thin AFDB-h5 reader/writer driver is operator code (heavy I/O). Output: a master record set
(`full_seq` + half-open coords). Write headers 1-based inclusive: `>{acc}_IDR_{start+1}-{end}`.

## 2. Cluster (mmseqs linclust, 90% id / 80% cov)

On the **full-length** sequences (dedups multi-IDR proteins to one representative):

```bash
mmseqs createdb  alldata_seqs.fasta  DB
mmseqs linclust  DB  CLU  tmp  --min-seq-id 0.9 --cov-mode 0 -c 0.8 --cluster-mode 2
mmseqs createsubdb CLU DB REPS && mmseqs convert2fasta REPS reps.fasta
```

## 3. Split + leakage filter (99 / 0.5 / 0.5)

```bash
# random split of reps.fasta -> train/val/test (99 / 0.5 / 0.5)
# leakage removal (ESM-2 style): drop train seqs >=50% id to any val/test/DisProt seq
mmseqs search COMBINED_DB TRAIN_DB hits tmp --min-seq-id 0.5 -c 0.8 --cov-mode 0 -s 7 --max-seqs 300
# remove the matched train accessions -> train_dedup
```

## 4. Write record FASTAs

For each split, write `{split}.fasta` of `>{acc}_IDR_{x}-{y}` (1-based inclusive) + `full_seq`,
**dropping any sequence longer than the length cap** and any with non-canonical residues
(`idiom.data.io.read_records` re-checks both on load, so this is belt-and-suspenders).

Output consumed directly by `idiom.data.datamodule.RecordDataModule(train_fasta=..., ...)`.
