#!/bin/bash
# Regenerate manuscript figures directly into the overleaf figs/ dir.
#
# Each figure script reads its experiment-output input(s) and writes
#   $IDIOM_FIG_DIR/<subdir>/<name>.{pdf,png}
# named to match the LaTeX \includegraphics path, so this overwrites in place and main.tex
# picks them up. After running, commit + push the OVERLEAF repo separately.
#
# Figure generation is light (CPU); run locally or `srun -c 4 --mem 8GB -t 00:30:00 bash bash/figures.bash`.
set -euo pipefail

REPO=/data2/scratch/jxliu2/idiom
source "$REPO/.venv/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO"   # so `analysis.*` (repo-only, not in the wheel) is importable

# Output root = the manuscript figs/ dir (override by exporting IDIOM_FIG_DIR yourself).
export IDIOM_FIG_DIR="${IDIOM_FIG_DIR:-/data2/scratch/jxliu2/papers/overleaf/IDiom-manuscript-v1/figs}"
echo "Writing figures to: $IDIOM_FIG_DIR"

# Scripts live under analysis/figures/<section>/ (see analysis/figures/README.md).
# --- pretraining-data figures (corpus_* read the master h5; corpus_composition reads the filtered FASTA) ---
python analysis/figures/pretraining_data/corpus_lengths.py
python analysis/figures/pretraining_data/corpus_plddt.py
python analysis/figures/pretraining_data/corpus_plddt_regions.py
python analysis/figures/pretraining_data/corpus_idr_fraction.py
python analysis/figures/pretraining_data/corpus_composition.py
python analysis/figures/pretraining_data/composition_vs_disprot.py
# --- add one line per figure as the other sections are ported in P6 ---
# python analysis/figures/disorder/disorder_barplot.py    --kind metapredict --fasta /path/gen.fasta
# python analysis/figures/biophysics/sparrow_metrics.py    --metric FCR       --pickles /path/*.pkl
# python analysis/figures/training_curves/training_curves.py --logs /path/lightning_logs
# python analysis/figures/sae/feature_analysis.py          --features /path/feature_dataset

echo "Done. Now: cd \"\$(dirname \"\$IDIOM_FIG_DIR\")\" && git add figs && git commit -m 'update figs' && git push"
