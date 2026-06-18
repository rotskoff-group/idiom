"""Record filters + driver: master h5 -> filtered record FASTA (curation stage 3).

Two pure, CPU-testable filters reduce the curated master (`AFDB_IDR_90_alldata.h5`) to the
training records. The legacy v1 path baked both into `make_AFDB_FIM.py` (see `_legacy_reference/`);
here they are standalone so the on-the-fly v2 store can apply them at write time.

  - length:           keep proteins whose `full_seq` fits the token budget (<= max_len - 4)
  - fully-low-pLDDT:  drop proteins with no folded region (fully disordered)

The heavy AFDB-h5 driver that maps these over the real corpus is operator code (the runbook).
"""

from __future__ import annotations

import numpy as np

from idiom.data.dataset import max_protein_len

from .extract import has_folded_segment


def passes_length(full_len: int, max_len: int = 1024) -> bool:
    """True iff a `full_seq` of `full_len` residues fits a `max_len`-token context.

    The cap is `max_len - 4` (3 FIM markers + START/STOP boundary), i.e. **<= 1020** for the
    default 1024-token model — the same bound the dataset enforces on load.
    """
    return full_len <= max_protein_len(max_len)


def is_fully_low_plddt(full_avg_plddt: np.ndarray, *, folded_thresh: float = 80.0) -> bool:
    """True iff the protein is fully disordered (no folded region) -> drop it.

    Uses the aggressive no-folded-segment criterion (see :func:`extract.has_folded_segment`):
    stricter than `max(full_avg_plddt) < {folded_thresh}`, since it also removes proteins whose
    only high-pLDDT residues form a sub-`min_seg_length` blip. `full_avg_plddt` is the window-15
    averaged per-residue pLDDT stored on each record (so no re-smoothing here: window=1).
    """
    return not has_folded_segment(full_avg_plddt, folded_thresh=folded_thresh, window=1)


# ── operator driver: curated master h5 -> filtered record FASTA ──────────────────────────────
# Heavy I/O (streams the 73M-row, ~125G master). Emits the record store consumed downstream:
# `>{base_acc}_IDR_{x}-{y}` (1-based inclusive) + full_seq, one line per kept IDR.

DEFAULT_MASTER = (
    "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/pretraining/AFDB/"
    "clustering_90/AFDB_IDR_90_alldata.h5"
)


def write_filtered_fasta(
    h5_path: str, out_path: str, *, max_len: int = 1024, chunk: int = 200_000
) -> tuple[int, int]:
    """Stream `h5_path`, apply both filters, write the record FASTA. Returns (kept, total).

    The master's `accession_ids` are `{base}_{s}-{e}` (e.g. `D2CY91-F1_0-149`); `idr_end` is
    0-based **inclusive**, so the 1-based inclusive header span is `(idr_start+1, idr_end+1)`.
    `full_length` and `full_avg_plddt` are per-protein, so the fully-low verdict is cached across
    a protein's (contiguous) IDR rows.
    """
    import h5py
    import numpy as np

    cap = max_protein_len(max_len)
    kept = total = 0
    last_base: str | None = None
    last_low = False
    with h5py.File(h5_path, "r") as f, open(out_path, "w") as out:
        n = f["accession_ids"].shape[0]
        d_acc, d_seq, d_fl = f["accession_ids"], f["full_seq"], f["full_length"]
        d_s, d_e, d_fap = f["idr_start"], f["idr_end"], f["full_avg_plddt"]
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            acc, seq, fl = d_acc[lo:hi], d_seq[lo:hi], d_fl[lo:hi]
            st, en, fap = d_s[lo:hi], d_e[lo:hi], d_fap[lo:hi]
            for j in range(hi - lo):
                total += 1
                if int(fl[j]) > cap:  # length filter (per-protein; applies to all its IDRs)
                    continue
                base = acc[j].decode().rsplit("_", 1)[0]
                if base != last_base:  # fully-low is per-protein; recompute only on change
                    last_base, last_low = base, is_fully_low_plddt(np.asarray(fap[j]))
                if last_low:
                    continue
                out.write(f">{base}_IDR_{int(st[j]) + 1}-{int(en[j]) + 1}\n{seq[j].decode()}\n")
                kept += 1
    return kept, total


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Stage 3: filter master h5 -> record FASTA.")
    ap.add_argument("--h5", default=DEFAULT_MASTER, help="curated master AFDB_IDR_90_alldata.h5")
    ap.add_argument("--out", required=True, help="output record FASTA")
    ap.add_argument("--max-len", type=int, default=1024, help="token budget; cap = max_len - 4")
    args = ap.parse_args()
    kept, total = write_filtered_fasta(args.h5, args.out, max_len=args.max_len)
    print(f"kept {kept:,} / {total:,} IDR records -> {args.out}")


if __name__ == "__main__":
    main()
