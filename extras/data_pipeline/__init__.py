"""Offline curation: AFDB -> IDR records -> clustered, split, leakage-filtered record FASTAs.

This is the one-time, offline pipeline that produces the training **record store** (per-split
``train/val/test.fasta`` in the ``_IDR_x-y`` convention, D16). It is *not* run during training
and is intentionally separate from the on-the-fly data path (``idiom.data.dataset``).

v2 changes vs the legacy `data_preprocess`:
- **No FIM/precompute materialization** — the store is raw ``full_seq`` + coords (Option A);
  FIM assembly + tokenization happen on the fly in the dataset.
- Protein-length cap is **≤ max_len − 4** (room for the 3 FIM markers + START), default 1020
  for a 1024-token model, replacing the legacy ``full_length ≤ 512``.

Stages (see ``RUNBOOK.md`` for the exact commands — the mmseqs/AFDB steps are heavy and
operator-run):

  1. extract  — pLDDT-based IDR segmentation from AFDB parts        (:func:`extract.extract_idrs`)
  2. cluster  — mmseqs linclust, 90% id / 80% cov                   (RUNBOOK)
  3. split    — random 99/0.5/0.5 + ≥50%-id leakage removal         (RUNBOOK)
  4. write    — per-split record FASTAs (`_IDR_x-y`, length-capped) (RUNBOOK)
"""
