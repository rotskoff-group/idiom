"""DisProt leakage dedup (curation stage 4).

Removes training IDR records that are >=50% identical to any DisProt IDR, so DisProt stays a
clean external benchmark. The comparison is **IDR-vs-IDR** (a short DisProt IDR vs a long full
protein never satisfies 80% coverage), so we search against the *IDR substrings* of the record
FASTA, then drop the matched records from the full-`full_seq` FASTA.

Pure helpers here (CPU-testable); the mmseqs search that joins query+target lives in `dedup.bash`.
CLI subcommands let that bash build the two FASTAs and apply the removal:

    python -m data_pipeline.dedup disprot-fasta --json DISPROT.json --out disprot_idrs.fasta
    python -m data_pipeline.dedup idr-fasta     --fasta records.fasta --out train_idrs.fasta
    python -m data_pipeline.dedup remove        --fasta records.fasta --remove hits_col2.txt --out kept.fasta
"""

from __future__ import annotations

import json
from pathlib import Path

from idiom.data.io import read_records

DEFAULT_DISPROT_JSON = (
    "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/reference/disprot/"
    "DisProt release_2025_06 with_ambiguous_evidences.json"
)


def disprot_idr_fasta(
    json_path: str, out_path: str, *, min_idr_length: int = 30, exclude_full_idps: bool = True
) -> int:
    """Write DisProt 'D' (disordered) consensus regions as an IDR FASTA. Returns count written."""
    data = json.loads(Path(json_path).read_text())
    entries = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
    n = 0
    with open(out_path, "w") as out:
        for entry in entries:
            acc, seq = entry.get("acc"), entry.get("sequence")
            states = entry.get("disprot_consensus", {}).get("Structural state", [])
            if not (acc and seq and isinstance(states, list)):
                continue
            for idx, region in enumerate(states):
                s, e = region.get("start"), region.get("end")
                if region.get("type") != "D" or not (isinstance(s, int) and isinstance(e, int)):
                    continue
                idr = seq[s - 1 : e]  # DisProt is 1-based inclusive
                if len(idr) < min_idr_length or (exclude_full_idps and len(idr) >= len(seq)):
                    continue
                out.write(f">{acc}_{idx}_{s}-{e}\n{idr}\n")
                n += 1
    return n


def write_idr_fasta(record_fasta: str, out_path: str) -> int:
    """Re-emit a record FASTA as IDR-only sequences, keyed by the *same* `_IDR_x-y` header token.

    So an mmseqs hit on this FASTA names exactly the record to drop. Returns count written.
    """
    n = 0
    with open(out_path, "w") as out:
        for r in read_records(record_fasta):
            out.write(f">{r.accession}_IDR_{r.idr_start + 1}-{r.idr_end}\n{r.full_seq[r.idr_start : r.idr_end]}\n")
            n += 1
    return n


def remove_headers(record_fasta: str, headers: set[str], out_path: str) -> tuple[int, int]:
    """Copy `record_fasta` to `out_path`, dropping entries whose header token is in `headers`.

    Returns (kept, removed). Header token = first whitespace field after '>'.
    """
    kept = removed = 0
    drop = False
    with open(record_fasta) as fin, open(out_path, "w") as out:
        for line in fin:
            if line.startswith(">"):
                drop = line[1:].split()[0] in headers
                if drop:
                    removed += 1
                else:
                    kept += 1
            if not drop:
                out.write(line)
    return kept, removed


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Stage 4: DisProt leakage dedup helpers.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("disprot-fasta", help="DisProt JSON -> IDR FASTA (search query)")
    p.add_argument("--json", default=DEFAULT_DISPROT_JSON)
    p.add_argument("--out", required=True)

    p = sub.add_parser("idr-fasta", help="record FASTA -> IDR-only FASTA (search target)")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("remove", help="drop records whose header is listed in --remove")
    p.add_argument("--fasta", required=True)
    p.add_argument("--remove", required=True, help="file of header tokens to drop (one per line)")
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "disprot-fasta":
        print(f"wrote {disprot_idr_fasta(args.json, args.out):,} DisProt IDRs -> {args.out}")
    elif args.cmd == "idr-fasta":
        print(f"wrote {write_idr_fasta(args.fasta, args.out):,} IDRs -> {args.out}")
    elif args.cmd == "remove":
        headers = {ln.strip() for ln in Path(args.remove).read_text().splitlines() if ln.strip()}
        kept, removed = remove_headers(args.fasta, headers, args.out)
        print(f"kept {kept:,}, removed {removed:,} -> {args.out}")


if __name__ == "__main__":
    main()
