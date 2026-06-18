# figures/ — downstream paper analysis & figures (not shipped)

Manuscript-reproduction code: evaluation (DisProt / CATH / AFDB-pLDDT>70 controls, DeepLoc,
catGRANULE, W1 / kappa / motif / disorder), interpretability analyses, and figure scripts. Imports
the `idiom` library; **not** part of the installed wheel (kept repo-only under `extras/`, run via
`figures.bash`). Reusable interpretability *methods* live in the library (`idiom.sae`); only
paper-specific analyses and figures live here.

Each figure script reads experiment output(s) and writes a PDF into the overleaf `figs/` tree via
`_style.save_fig(fig, name, subdir=...)`, where **the output path = the LaTeX `\includegraphics`
path**, so regenerating overwrites in place. Run with the repo root on `PYTHONPATH` (so `extras.*` is
importable) and `IDIOM_FIG_DIR` set (see `figures.bash`):

```bash
export PYTHONPATH=/path/to/idiom        # repo root (so `extras.figures.*` is importable)
export IDIOM_FIG_DIR=/path/to/manuscript/figs
export IDIOM_DATA=/path/to/idiom_data   # curated corpus + reference sets
python -m extras.figures.pretraining_data.corpus_plddt --n 200000
```

## Shared (top level)
- `_style.py` — `use_style()`, `save_fig()`, `COLORS` / `DIVERGING` / `SEQUENTIAL`.
- `idiom.mplstyle` — the manuscript matplotlib style (Liberation Sans).

## Sections (script dir → output subdir)
| dir | writes to | content |
|---|---|---|
| `pretraining_data/` | `si_figs/dataset/` | AFDB corpus characterization (lengths, pLDDT, IDR fraction) |
| `condensate_rl/` | `journal_figs/` | ProtGPS-GRPO validation (specificity, DeepLoc/catGRANULE, naturalness) |
| `disorder/` | `si_figs/disorder/` | secondary-structure + disorder-predictor (metapredict, IUPred3) |
| `biophysics/` | `si_figs/biophysics/` | SPARROW sequence metrics (FCR, κ, SHD, …) |
| `training_curves/` | `si_figs/curves/` | pretraining + RL training curves |
| `sae/` | `si_figs/sae/` | sparse-autoencoder feature analyses |
| `generation/` | (main / various) | generated-sequence analyses |

### `pretraining_data/`
All `corpus_*` and `composition_*` figures (SI S2–S7) characterize the **final training corpus**
(`intermediate/AFDB_IDR_90_len1020_rm_full_low_plddt_dedup_disprot_signalp.fasta`, 54.2M IDRs after
the length/pLDDT filter, DisProt dedup, and signal-peptide trim). Sequence-derived figures read the
FASTA directly; the pLDDT figures join each sampled record back to the extraction master h5 via
`final_corpus.plddt_by_protein` (`final_corpus.py` holds the shared FASTA/h5 paths + the join).
- `corpus_lengths.py` — IDR + protein length distributions, vs DisProt.
- `corpus_plddt.py` — mean IDR pLDDT + max protein pLDDT (confirms the fully-low-pLDDT filter).
- `corpus_plddt_regions.py` — IDR vs non-IDR region pLDDT (per-protein, per-residue, joint).
- `corpus_idr_fraction.py` — IDR fraction of each protein.
- `corpus_composition.py` — AA composition + log2 enrichment, IDR vs non-IDR.
- `composition_vs_disprot.py` — training IDRs vs DisProt IDRs as enrichment over the CATH baseline.
- `curation_funnel.py` — record-count funnel across all curation stages.
- `signalp_effect.py` — compositional effect of the stage-4b signal-peptide (SignalP-6) filter.

### `condensate_rl/`
Validation of the ProtGPS-reward GRPO runs (see the 2026-06-18 lab-journal entry). Reads the
condensate-RL results dir (`$IDIOM_CONDENSATE_RESULTS`); writes to `journal_figs/`.
- `specificity_matrix.py` — ProtGPS model×compartment heatmap (target-selectivity / diagonal dominance).
- `external_validation.py` — DeepLoc (organelle) + catGRANULE (LLPS) per-target validation vs base.
- `naturalness.py` — composition / complexity / entropy / perplexity vs natural IDRs (no reward hacking).
