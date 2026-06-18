# Manuscript figures

Each script reads experiment output(s) and writes a PDF into the overleaf `figs/` tree via
`_style.save_fig(fig, name, subdir=...)`, where **the output path = the LaTeX `\includegraphics`
path**, so regenerating overwrites in place. Run with the repo on `PYTHONPATH` and `IDIOM_FIG_DIR`
set (see `bash/figures.bash`):

```bash
export PYTHONPATH=/path/to/idiom        # repo root (so `analysis.*` is importable)
export IDIOM_FIG_DIR=/path/to/manuscript/figs
export IDIOM_DATA=/path/to/idiom_data   # curated corpus + reference sets
python analysis/figures/pretraining_data/corpus_plddt.py --n 200000
```

## Shared (top level)
- `_style.py` — `use_style()`, `save_fig()`, `COLORS` / `DIVERGING` / `SEQUENTIAL`.
- `idiom.mplstyle` — the manuscript matplotlib style (Liberation Sans).

## Sections (script dir → SI output subdir)
| dir | writes to `figs/si_figs/...` | content |
|---|---|---|
| `pretraining_data/` | `dataset/` | AFDB corpus characterization (lengths, pLDDT, IDR fraction) |
| `disorder/` | `disorder/` | secondary-structure + disorder-predictor (metapredict, IUPred3) |
| `biophysics/` | `biophysics/` | SPARROW sequence metrics (FCR, κ, SHD, …) |
| `training_curves/` | `curves/` | pretraining + RL training curves |
| `sae/` | `sae/` | sparse-autoencoder feature analyses |
| `generation/` | (main / various) | generated-sequence analyses |

### `pretraining_data/` (current)
All `corpus_*` and `composition_*` figures (SI S2–S7) characterize the **final training corpus**
(`intermediate/AFDB_IDR_90_len1020_rm_full_low_plddt_dedup_disprot_signalp.fasta`, 54.2M IDRs after
the length/pLDDT filter, DisProt dedup, and signal-peptide trim). Sequence-derived figures read the
FASTA directly; the pLDDT figures join each sampled record back to the extraction master h5 for
per-residue pLDDT via `final_corpus.plddt_by_protein` (`final_corpus.py` holds the shared FASTA/h5
paths + the join). Run from the repo root with `PYTHONPATH`/`IDIOM_FIG_DIR` set (heavy reads → via
`srun`).
- `corpus_lengths.py` — IDR + protein length distributions, vs DisProt.
- `corpus_plddt.py` — mean IDR pLDDT + max protein pLDDT (confirms the fully-low-pLDDT filter).
- `corpus_plddt_regions.py` — IDR vs non-IDR region pLDDT (per-protein, per-residue, joint).
- `corpus_idr_fraction.py` — IDR fraction of each protein (single panel; no whole-protein records remain).
- `corpus_composition.py` — AA composition + log2 enrichment, IDR vs non-IDR.

`composition_vs_disprot.py` — training IDRs vs DisProt IDRs as relative enrichment over the CATH
baseline (`reference/cath/`, `reference/disprot/`). Replicates the v1
`figure_scripts/idps_dp_idrs/composition.py` analysis (training/DisProt only; no generated sets yet).

`curation_funnel.py` — record-count funnel across all curation stages (raw 110M → 90% reps 73M →
length+pLDDT 57.8M → DisProt dedup 57.7M → signal-peptide filter 54.2M). Counts are measured
provenance constants (no data read); self-contained.

`signalp_effect.py` — compositional effect of the stage-4b signal-peptide (SignalP-6) filter,
comparing the corpus before (`..._dedup_disprot.fasta`) and after (`..._dedup_disprot_signalp.fasta`)
the filter: N-terminal hydropathy profile (the signal-peptide hydrophobic bump being removed) +
per-residue log2(before/after) composition shift.
