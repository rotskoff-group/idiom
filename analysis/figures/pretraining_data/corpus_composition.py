"""SI figure: amino-acid composition of IDR vs non-IDR regions of the filtered training corpus.

Runs on the curation stage-3 output FASTA (length + fully-low-pLDDT filtered, `_IDR_x-y` records).
Records are grouped by protein so a protein's other IDRs are excluded from its non-IDR residues.
Left: per-residue AA frequency, IDR vs non-IDR (ordered/flanking). Right: log2 enrichment
(IDR / non-IDR), the compositional signature of disorder. AAs sorted by enrichment.

  python analysis/figures/pretraining_data/corpus_composition.py --fasta .../AFDB_IDR_90_len1020_rm_full_low_plddt.fasta --n 200000
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter

import numpy as np

from analysis.figures._style import COLORS, save_fig, use_style
from idiom.data.io import parse_idr_header

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def iter_records(path):
    """Lazily stream `(accession, full_seq, idr_start, idr_end)` from a record FASTA.

    Hand-rolled rather than `idiom.data.io.read_records`, which eagerly materializes the whole
    (27 GB) file before any sampling.
    """
    header = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line[1:]
            elif header is not None:
                acc, start, end = parse_idr_header(header)
                yield acc, line, start, end
                header = None
DEFAULT_FASTA = (
    "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/pretraining/AFDB/"
    "intermediate/AFDB_IDR_90_len1020_rm_full_low_plddt.fasta"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--n", type=int, default=50_000, help="records to sample (from the start)")
    ap.add_argument("--name", default="corpus_composition")
    args = ap.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    idr_counts: Counter = Counter()
    non_counts: Counter = Counter()
    # group consecutive records by protein (records stay protein-contiguous pre-split)
    records = itertools.islice(iter_records(args.fasta), args.n)
    for _, group in itertools.groupby(records, key=lambda r: r[0]):
        seq, spans = None, []
        for acc, full_seq, start, end in group:
            seq = full_seq
            spans.append((start, end))
        spans.sort()
        idr_counts.update("".join(seq[s:e] for s, e in spans))  # C-level char tally
        prev, non_parts = 0, []
        for s, e in spans:  # the gaps between IDR spans = non-IDR (ordered/flanking) residues
            non_parts.append(seq[prev:s])
            prev = e
        non_parts.append(seq[prev:])
        non_counts.update("".join(non_parts))

    idr_tot = sum(idr_counts[a] for a in RESIDUES)
    non_tot = sum(non_counts[a] for a in RESIDUES)
    idr_freq = np.array([idr_counts[a] / idr_tot for a in RESIDUES])
    non_freq = np.array([non_counts[a] / non_tot for a in RESIDUES])
    enrich = np.log2(idr_freq / non_freq)

    order = np.argsort(enrich)[::-1]  # most IDR-enriched first
    aas = np.array(list(RESIDUES))[order]
    x = np.arange(len(RESIDUES))
    blue, orange = COLORS["darkblue"], COLORS["orange"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))

    ax1.bar(x - 0.2, idr_freq[order] * 100, width=0.4, color=blue, label="IDR")
    ax1.bar(x + 0.2, non_freq[order] * 100, width=0.4, color=orange, label="non-IDR")
    ax1.set_xticks(x)
    ax1.set_xticklabels(aas)
    ax1.set_xlabel("amino acid")
    ax1.set_ylabel("frequency (%)")
    ax1.set_title("composition", fontsize=18)
    ax1.legend(frameon=False)

    colors = [blue if e > 0 else orange for e in enrich[order]]
    ax2.bar(x, enrich[order], color=colors)
    ax2.axhline(0, color=COLORS["black"], lw=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(aas)
    ax2.set_xlabel("amino acid")
    ax2.set_ylabel(r"$\log_2$(IDR / non-IDR)")
    ax2.set_title("enrichment in IDRs", fontsize=18)

    fig.suptitle(f"filtered training corpus (n = {idr_tot + non_tot:,} residues)", fontsize=16)
    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
