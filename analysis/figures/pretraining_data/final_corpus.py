"""Shared access to the FINAL training corpus for the SI dataset figures (S2–S7).

The final corpus FASTA --- ``AFDB_IDR_90_len1020_rm_full_low_plddt_dedup_disprot_signalp.fasta``,
54.2M records after the length/pLDDT filter, DisProt dedup, and SignalP signal-peptide trim --- is
the exact set the model is trained on. It carries only sequences and ``_IDR_x-y`` headers, so
sequence-derived figures (lengths, IDR fraction, composition) read it directly via
:func:`iter_records`. The pLDDT figures need per-residue pLDDT, which lives only in the extraction
master h5; :func:`plddt_by_protein` joins each sampled final record back to the master by protein
accession (every final protein descends from the master). Records are protein-contiguous and both
files are protein-sorted, so the join early-exits after the first chunk for a head sample.
"""

from __future__ import annotations

import itertools

import h5py
import numpy as np

from analysis.figures.pretraining_data.corpus_composition import iter_records

DATA = "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data"
INTER = f"{DATA}/pretraining/AFDB/intermediate"
FINAL_FASTA = f"{INTER}/AFDB_IDR_90_len1020_rm_full_low_plddt_dedup_disprot_signalp.fasta"
MASTER_H5 = f"{DATA}/pretraining/AFDB/clustering_90/AFDB_IDR_90_alldata.h5"


def sample_records(fasta: str, n: int):
    """First ``n`` final-corpus records as ``(acc, full_seq, start, end)`` (0-based half-open IDR)."""
    return list(itertools.islice(iter_records(fasta), n))


def plddt_by_protein(accs, h5: str = MASTER_H5, chunk: int = 2_000_000) -> dict:
    """Map each protein accession in ``accs`` to its per-residue ``full_avg_plddt`` array.

    Scans the master h5 ``accession_ids`` in chunks, recording the first row per needed protein and
    stopping once all are found, then reads those rows' ``full_avg_plddt`` (one fancy-indexed read).
    """
    needed = set(accs)
    first_row: dict[str, int] = {}
    with h5py.File(h5, "r") as f:
        ds = f["accession_ids"]
        ntot = ds.shape[0]
        for s in range(0, ntot, chunk):
            for j, a in enumerate(ds[s : s + chunk]):
                b = a.decode().rsplit("_", 1)[0]
                if b in needed and b not in first_row:
                    first_row[b] = s + j
            if len(first_row) == len(needed):
                break
        if len(first_row) < len(needed):
            print(f"  WARNING: {len(needed) - len(first_row):,} proteins not found in master h5")
        items = sorted(first_row.items(), key=lambda kv: kv[1])  # ascending row idx for h5 fancy index
        plddt = f["full_avg_plddt"][[i for _, i in items]]
    return {b: np.asarray(p, dtype=np.float32) for (b, _), p in zip(items, plddt)}
