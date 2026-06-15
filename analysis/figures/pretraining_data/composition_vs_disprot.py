"""SI figure: amino-acid composition of training IDRs vs DisProt IDRs, relative to a CATH baseline.

Replicates the (non-generated part of the) v1 `idps_dp_idrs/composition.py` "relative enrichment"
analysis for the new filtered training corpus: each set's per-residue AA frequency is divided by
the CATH folded-domain frequency, so 1.0 = CATH and bars above/below show enrichment/depletion in
disordered sequence. Training IDRs are the IDR substrings of the filtered record FASTA; DisProt IDRs
are the experimentally-annotated disordered ('D') consensus regions (non-IDP).

  python analysis/figures/pretraining_data/composition_vs_disprot.py --n 50000
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.figures._style import COLORS, save_fig, use_style
from analysis.figures.pretraining_data.corpus_composition import iter_records

RES = "ACDEFGHIKLMNPQRSTVWY"
DATA = "/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data"
DEFAULT_TRAIN = f"{DATA}/pretraining/AFDB/intermediate/AFDB_IDR_90_len1020_rm_full_low_plddt.fasta"
DEFAULT_DISPROT = f"{DATA}/reference/disprot/DisProt release_2025_06 with_ambiguous_evidences.json"
DEFAULT_CATH = f"{DATA}/reference/cath/cath-domain-seqs-S60_1000.fa"


def composition(seqs) -> np.ndarray:
    """Per-residue frequency over the 20 canonical AAs (total includes any other chars, as in v1)."""
    counts: Counter = Counter()
    total = 0
    for s in seqs:
        counts.update(s)
        total += len(s)
    return np.array([counts[a] / total for a in RES]) if total else np.zeros(len(RES))


def disprot_idrs(json_path, min_idr_length=30, max_seq_length=1020):
    """DisProt 'D' (disordered) consensus regions: idr length>=min, full seq<=max, non-IDP.

    `max_seq_length=1020` matches the training-corpus protein-length cap (was 512 in v1).
    """
    data = json.loads(Path(json_path).read_text())
    entries = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
    out = []
    for e in entries:
        acc, seq = e.get("acc"), e.get("sequence")
        if not (acc and seq) or len(seq) > max_seq_length:
            continue
        for r in e.get("disprot_consensus", {}).get("Structural state", []):
            s, en = r.get("start"), r.get("end")
            if r.get("type") != "D" or not (isinstance(s, int) and isinstance(en, int)):
                continue
            idr = seq[s - 1 : en]
            if len(idr) >= min_idr_length and len(idr) < len(seq):
                out.append(idr)
    return out


def fasta_seqs(path, min_len=30, max_len=512):
    seqs, cur = [], ""
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if min_len <= len(cur) <= max_len:
                    seqs.append(cur)
                cur = ""
            else:
                cur += line.strip()
    if min_len <= len(cur) <= max_len:
        seqs.append(cur)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-fasta", default=DEFAULT_TRAIN)
    ap.add_argument("--disprot-json", default=DEFAULT_DISPROT)
    ap.add_argument("--cath", default=DEFAULT_CATH)
    ap.add_argument("--n", type=int, default=50_000, help="training records to sample")
    ap.add_argument("--disprot-max-len", type=int, default=1020, help="DisProt full-seq cap (match corpus)")
    ap.add_argument("--disprot-min-idr", type=int, default=30, help="DisProt min IDR length")
    ap.add_argument("--name", default="composition_vs_disprot")
    args = ap.parse_args()

    use_style()
    import matplotlib.pyplot as plt

    train_idr = [seq[s:e] for _, seq, s, e in itertools.islice(iter_records(args.train_fasta), args.n)]
    dp_idr = disprot_idrs(args.disprot_json, args.disprot_min_idr, args.disprot_max_len)
    cath = fasta_seqs(args.cath)
    print(f"train IDRs={len(train_idr):,}  DisProt IDRs={len(dp_idr):,}  CATH={len(cath):,}")

    cath_f = composition(cath)
    train_e = composition(train_idr) / cath_f  # fold enrichment relative to CATH
    dp_e = composition(dp_idr) / cath_f

    order = np.argsort(train_e)[::-1]  # most training-enriched first
    aas = np.array(list(RES))[order]
    x = np.arange(len(RES))

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axhline(1.0, color=COLORS["black"], ls="--", lw=1.2, label="CATH reference")
    ax.bar(x - 0.2, train_e[order], width=0.4, color=COLORS["darkblue"], label="training IDRs")
    ax.bar(x + 0.2, dp_e[order], width=0.4, color=COLORS["green"], label="DisProt IDRs")
    ax.set_xticks(x)
    ax.set_xticklabels(aas)
    ax.set_xlabel("amino acid")
    ax.set_ylabel("relative enrichment")
    ax.margins(x=0.01)
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    print("saved", save_fig(fig, args.name, subdir="si_figs/dataset"))


if __name__ == "__main__":
    main()
