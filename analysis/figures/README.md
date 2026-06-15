# Manuscript figures

Each script reads experiment output(s) and writes a PDF into the overleaf `figs/` tree via
`_style.save_fig(fig, name, subdir=...)`, where **the output path = the LaTeX `\includegraphics`
path**, so regenerating overwrites in place. Run with the repo on `PYTHONPATH` and `IDIOM_FIG_DIR`
set (see `bash/figures.bash`):

```bash
export PYTHONPATH=/data2/scratch/jxliu2/idiom
export IDIOM_FIG_DIR=/data2/scratch/jxliu2/papers/overleaf/IDiom-manuscript-v1/figs
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
`corpus_*` read the curated master `clustering_90/AFDB_IDR_90_alldata.h5`:
- `corpus_lengths.py` — IDR + protein length distributions (Fig S1).
- `corpus_plddt.py` — mean IDR pLDDT + max protein pLDDT (Fig S2).
- `corpus_plddt_regions.py` — IDR vs non-IDR region pLDDT (per-protein, per-residue, joint).
- `corpus_idr_fraction.py` — IDR fraction of each protein (analogue of legacy `pct_idr.py`).

`corpus_composition.py` reads the **filtered** record FASTA
(`intermediate/AFDB_IDR_90_len1020_rm_full_low_plddt.fasta`) — AA composition + enrichment,
IDR vs non-IDR.

`composition_vs_disprot.py` — training IDRs vs DisProt IDRs as relative enrichment over the CATH
baseline (`reference/cath/`, `reference/disprot/`). Replicates the v1
`figure_scripts/idps_dp_idrs/composition.py` analysis (training/DisProt only; no generated sets yet).
