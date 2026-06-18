"""SI figure: compositional effect of the stage-4b signal-peptide (SignalP-6) filter.

Compares the corpus immediately *before* the signal-peptide filter (the DisProt-dedup FASTA) with
the corpus *after* it (the final training FASTA). Signal peptides are short, strongly hydrophobic
N-terminal stretches that AlphaFold's pLDDT segmentation mislabels as IDRs; the filter trims/drops
the records whose IDR overlaps a SignalP cleavage site. The figure isolates what that removed.

Left: mean Kyte--Doolittle hydropathy as a function of position from the IDR start, restricted to
**N-terminal IDRs** (header span starts at residue 1). Before filtering these carry a hydrophobic
bump over the first ~20 residues (the signal peptide); after filtering the trimmed/dropped records
are gone and the curve flattens to the disordered (hydrophilic) baseline. Right: per-residue
log2(before / after) over all sampled IDR residues --- the residues the filter depletes are the
order-promoting hydrophobics (L, A, V, F, I) characteristic of signal peptides.

  python analysis/figures/pretraining_data/signalp_effect.py --n 300000 --max-pos 40
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections import Counter

import numpy as np

from analysis.figures._style import COLORS, row_fig, save_fig, use_style
from analysis.figures.pretraining_data.corpus_composition import iter_records

RES = "ACDEFGHIKLMNPQRSTVWY"
# Kyte--Doolittle hydropathy (positive = hydrophobic).
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "V": 4.2, "Y": -1.3}

DATA = os.environ.get("IDIOM_DATA", "/path/to/idiom_data")
INTER = f"{DATA}/pretraining/AFDB/intermediate"
DEFAULT_PRE = f"{INTER}/AFDB_IDR_90_len1020_rm_full_low_plddt_dedup_disprot.fasta"
DEFAULT_POST = f"{INTER}/AFDB_IDR_90_len1020_rm_full_low_plddt_dedup_disprot_signalp.fasta"


def scan(path, n, max_pos):
    """One streaming pass: per-residue composition + N-terminal hydropathy profile.

    Returns (comp[20], profile_sum[max_pos], profile_cnt[max_pos]). ``profile`` accumulates KD
    hydropathy per position only over IDRs whose header span starts at residue 1 (start == 0).
    """
    comp: Counter = Counter()
    psum = np.zeros(max_pos)
    pcnt = np.zeros(max_pos)
    for _acc, full, s, e in itertools.islice(iter_records(path), n):
        idr = full[s:e]
        comp.update(idr)
        if s == 0:  # N-terminal IDR — where signal peptides live
            for i, aa in enumerate(idr[:max_pos]):
                if aa in KD:
                    psum[i] += KD[aa]
                    pcnt[i] += 1
    tot = sum(comp[a] for a in RES)
    freq = np.array([comp[a] / tot for a in RES]) if tot else np.zeros(len(RES))
    return freq, psum, pcnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=DEFAULT_PRE, help="corpus before the signal-peptide filter")
    ap.add_argument("--post", default=DEFAULT_POST, help="corpus after the signal-peptide filter")
    ap.add_argument("--n", type=int, default=300_000, help="records to sample from each corpus")
    ap.add_argument("--max-pos", type=int, default=40, help="positions from IDR start for profile")
    ap.add_argument("--name", default="signalp_effect")
    args = ap.parse_args()

    use_style()

    pre_f, pre_s, pre_c = scan(args.pre, args.n, args.max_pos)
    post_f, post_s, post_c = scan(args.post, args.n, args.max_pos)
    pre_prof = pre_s / np.maximum(pre_c, 1)
    post_prof = post_s / np.maximum(post_c, 1)
    print(f"N-terminal IDRs sampled: pre {int(pre_c[0]):,}  post {int(post_c[0]):,}")

    fig, (ax1, ax2) = row_fig(2)

    x = np.arange(args.max_pos) + 1
    ax1.axhline(0, color=COLORS["black"], lw=1.0, ls=":")
    ax1.plot(x, pre_prof, color=COLORS["red"], lw=2.5, label="before filter")
    ax1.plot(x, post_prof, color=COLORS["darkblue"], lw=2.5, label="after filter")
    ax1.set_xlabel("position from IDR N-terminus")
    ax1.set_ylabel("mean Kyte–Doolittle hydropathy")
    ax1.set_title("N-terminal IDRs")
    ax1.legend(frameon=False)

    lr = np.log2(pre_f / post_f)
    order = np.argsort(lr)[::-1]  # most depleted-by-filter first
    aas = np.array(list(RES))[order]
    xb = np.arange(len(RES))
    colors = [COLORS["red"] if v > 0 else COLORS["darkblue"] for v in lr[order]]
    ax2.bar(xb, lr[order], color=colors)
    ax2.axhline(0, color=COLORS["black"], lw=1.0)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(aas)
    ax2.set_xlabel("amino acid")
    ax2.set_ylabel(r"$\log_2$(before / after)")
    ax2.set_title("composition removed by filter")

    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
