"""SignalP signal-peptide filter (curation stage 4b).

The pLDDT segmentation mislabels N-terminal signal peptides (flexible, low-confidence in AlphaFold)
as IDRs: ~14% of records have an IDR overlapping a signal peptide. SignalP 6 (fast, GPU-converted)
flags them with a cleavage site (CS); the signal occupies residues ``1..CS`` (0-based ``[0, cs)``).
A record overlaps the signal iff its IDR ``start < cs``.

This module is pure/CPU-testable: it builds SignalP's input (one entry per parent protein), shards
it, and applies the output. The heavy GPU run is orchestrated by `signalp_filter.bash`. Two apply
modes:
  * ``drop``  — remove the whole record.
  * ``trim``  — cut the signal off the front (``IDR_{cs+1}-y``, full_seq kept as context), keeping
    the genuine mature remainder; drop only if < ``min_len`` remains. Biologically the cleaner
    choice: the cleavage site is the mature N-terminus, so the trimmed span is the real IDR.

Confidence: SignalP reports a probability per class plus a CS probability; we keep only calls with
predicted-class prob >= ``min_prob`` (and CS prob >= ``min_cs_prob``).

    python -m data_pipeline.signalp_filter protein-fasta --fasta records.fasta --out proteins.fasta
    python -m data_pipeline.signalp_filter shard        --fasta proteins.fasta --n 16 --out-prefix shards/shard
    python -m data_pipeline.signalp_filter apply        --record-fasta records.fasta --results r_*/prediction_results.txt \
        --mode trim --min-prob 0.9 --out kept.fasta
"""

from __future__ import annotations

import re
from pathlib import Path

from idiom.data.io import parse_idr_header

SIGNAL_TYPES = ("SP", "LIPO", "TAT", "TATLIPO", "PILIN")
# prediction_results.txt column index of each class's probability.
_PROB_COL = {"OTHER": 2, "SP": 3, "LIPO": 4, "TAT": 5, "TATLIPO": 6, "PILIN": 7}
_CS_RE = re.compile(r"CS pos:\s*(\d+)-\d+\.\s*Pr:\s*([\d.]+)")


def protein_fasta(record_fasta: str, out_path: str) -> int:
    """Record FASTA -> one entry per **unique parent protein** (SignalP input). Streams; returns n."""
    seen: set[str] = set()
    n = 0
    with open(record_fasta) as fin, open(out_path, "w") as out:
        acc, seq = None, []
        def emit():
            nonlocal n
            if acc is not None and acc not in seen:
                seen.add(acc)
                out.write(f">{acc}\n{''.join(seq)}\n")
                n += 1
        for line in fin:
            if line.startswith(">"):
                emit()
                try:
                    acc = parse_idr_header(line[1:])[0]
                except ValueError:
                    acc = None
                seq = []
            else:
                seq.append(line.strip())
        emit()
    return n


def shard_fasta(fasta: str, n: int, out_prefix: str) -> int:
    """Round-robin a 2-line FASTA into ``n`` balanced shards ``{out_prefix}_{i}.fasta``. Returns n records."""
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    outs = [open(f"{out_prefix}_{i}.fasta", "w") for i in range(n)]
    i, cur = 0, None
    try:
        with open(fasta) as fin:
            for line in fin:
                if line.startswith(">"):
                    cur = i % n
                    i += 1
                    outs[cur].write(line)
                elif cur is not None:
                    outs[cur].write(line)
    finally:
        for o in outs:
            o.close()
    return i


def signal_cleavage_sites(
    results_paths: list[str], *, types: tuple[str, ...] = SIGNAL_TYPES,
    min_prob: float = 0.9, min_cs_prob: float = 0.0,
) -> dict[str, int]:
    """Parse `prediction_results.txt` files -> ``{protein_acc: cleavage_site}`` for confident signals."""
    keep = set(types)
    sites: dict[str, int] = {}
    for path in results_paths:
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                c = line.rstrip("\n").split("\t")
                pred = c[1]
                if pred not in keep or float(c[_PROB_COL[pred]]) < min_prob:
                    continue
                m = _CS_RE.search(c[-1])
                if not m or float(m.group(2)) < min_cs_prob:
                    continue
                sites[c[0]] = int(m.group(1))
    return sites


def apply_filter(
    record_fasta: str, sites: dict[str, int], out_path: str, *,
    mode: str = "trim", min_len: int = 30, remove_list: str | None = None,
) -> tuple[int, int, int]:
    """Stream `record_fasta`, dropping/trimming records whose IDR overlaps the parent's signal.

    Returns (kept_unchanged, trimmed, dropped). `kept_unchanged` includes records of signal-free
    proteins and N-terminal-clear IDRs. For ``trim``, a record whose mature remainder < `min_len`
    is dropped. Streams a (possibly multi-line) record FASTA; emits single-line seqs.
    """
    kept = trimmed = dropped = 0
    removed = open(remove_list, "w") if remove_list else None

    def handle(out, token, seq):
        nonlocal kept, trimmed, dropped
        try:
            acc, start, end = parse_idr_header(token)
        except ValueError:
            out.write(f">{token}\n{seq}\n"); kept += 1; return
        cs = sites.get(acc)
        if cs is None or start >= cs:          # no signal, or IDR starts after the signal
            out.write(f">{token}\n{seq}\n"); kept += 1; return
        if mode == "drop":
            dropped += 1
            if removed: removed.write(f"{token}\n")
            return
        new_start = cs                          # mature N-terminus = cleavage site (0-based)
        if end - new_start >= min_len:
            out.write(f">{acc}_IDR_{new_start + 1}-{end}\n{seq}\n"); trimmed += 1
        else:
            dropped += 1
            if removed: removed.write(f"{token}\n")

    with open(record_fasta) as fin, open(out_path, "w") as out:
        token, seq = None, []
        for line in fin:
            if line.startswith(">"):
                if token is not None:
                    handle(out, token, "".join(seq))
                token, seq = line[1:].split()[0], []
            else:
                seq.append(line.strip())
        if token is not None:
            handle(out, token, "".join(seq))
    if removed:
        removed.close()
    return kept, trimmed, dropped


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Stage 4b: SignalP signal-peptide filter.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("protein-fasta", help="record FASTA -> unique parent-protein FASTA")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("shard", help="round-robin a FASTA into N shards")
    p.add_argument("--fasta", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out-prefix", required=True)

    p = sub.add_parser("apply", help="apply SignalP results: drop/trim records overlapping a signal")
    p.add_argument("--record-fasta", required=True)
    p.add_argument("--results", required=True, nargs="+", help="prediction_results.txt file(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("drop", "trim"), default="trim")
    p.add_argument("--types", nargs="+", default=list(SIGNAL_TYPES))
    p.add_argument("--min-prob", type=float, default=0.9, help="min predicted-class probability")
    p.add_argument("--min-cs-prob", type=float, default=0.0, help="min cleavage-site probability")
    p.add_argument("--min-len", type=int, default=30, help="trim: keep remainder only if >= this")
    p.add_argument("--remove-list", help="optional path to write dropped header tokens")

    args = ap.parse_args()
    if args.cmd == "protein-fasta":
        print(f"wrote {protein_fasta(args.fasta, args.out):,} proteins -> {args.out}")
    elif args.cmd == "shard":
        print(f"sharded {shard_fasta(args.fasta, args.n, args.out_prefix):,} records into {args.n}")
    elif args.cmd == "apply":
        sites = signal_cleavage_sites(
            args.results, types=tuple(args.types), min_prob=args.min_prob, min_cs_prob=args.min_cs_prob)
        kept, trimmed, dropped = apply_filter(
            args.record_fasta, sites, args.out, mode=args.mode, min_len=args.min_len,
            remove_list=args.remove_list)
        total = kept + trimmed + dropped
        print(f"signal proteins: {len(sites):,}  |  kept {kept:,}  trimmed {trimmed:,}  "
              f"dropped {dropped:,}  ({100*(trimmed+dropped)/max(total,1):.1f}% affected) -> {args.out}")


if __name__ == "__main__":
    main()
