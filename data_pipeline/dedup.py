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

from pathlib import Path

from idiom.data.io import read_records

from data_pipeline.disprot import DEFAULT_DISPROT_JSON, disprot_idr_records


def disprot_idr_fasta(
    json_path: str, out_path: str, *, min_idr_length: int = 30, max_seq_length: int = 1020
) -> int:
    """Write the benchmark DisProt IDRs as the dedup query FASTA. Returns count written.

    Uses the canonical parser (`data_pipeline.disprot`) so the query is exactly the DisProt IDR
    set the figures/benchmark use: 'D' regions, idr >= min, full seq <= max (1020 = corpus cap),
    full IDPs removed. Headers keep the `{acc}_{idx}_{start}-{end}` token.
    """
    records = disprot_idr_records(json_path, min_idr_length=min_idr_length, max_seq_length=max_seq_length)
    with open(out_path, "w") as out:
        for acc, idx, s, e, idr, _full in records:
            out.write(f">{acc}_{idx}_{s}-{e}\n{idr}\n")
    return len(records)


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
    p.add_argument("--max-seq-len", type=int, default=1020, help="full-seq cap (match corpus)")
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
        n = disprot_idr_fasta(args.json, args.out, max_seq_length=args.max_seq_len)
        print(f"wrote {n:,} DisProt IDRs -> {args.out}")
    elif args.cmd == "idr-fasta":
        print(f"wrote {write_idr_fasta(args.fasta, args.out):,} IDRs -> {args.out}")
    elif args.cmd == "remove":
        headers = {ln.strip() for ln in Path(args.remove).read_text().splitlines() if ln.strip()}
        kept, removed = remove_headers(args.fasta, headers, args.out)
        print(f"kept {kept:,}, removed {removed:,} -> {args.out}")


if __name__ == "__main__":
    main()
