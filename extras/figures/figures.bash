#!/bin/bash
# Regenerate manuscript figures directly into the overleaf figs/ dir.
#
# Each figure script reads its experiment-output input(s) and writes
#   $IDIOM_FIG_DIR/<subdir>/<name>.{pdf,png}
# named to match the LaTeX \includegraphics path, so this overwrites in place and main.tex
# picks them up. After running, commit + push the OVERLEAF repo separately.
#
# Figure generation is light (CPU); run locally or `srun -c 4 --mem 8GB -t 00:30:00 bash extras/figures/figures.bash`.
set -euo pipefail

REPO="${IDIOM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/.venv/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO"   # so `extras.figures.*` (repo-only, not in the wheel) is importable

# Output root = the manuscript figs/ dir (override by exporting IDIOM_FIG_DIR yourself).
export IDIOM_FIG_DIR="${IDIOM_FIG_DIR:-/path/to/manuscript/figs}"
echo "Writing figures to: $IDIOM_FIG_DIR"

# Scripts live under extras/figures/<section>/ (see extras/figures/README.md).
# --- pretraining-data figures (corpus_* read the master h5; corpus_composition reads the filtered FASTA) ---
python -m extras.figures.pretraining_data.corpus_lengths
python -m extras.figures.pretraining_data.corpus_plddt
python -m extras.figures.pretraining_data.corpus_plddt_regions
python -m extras.figures.pretraining_data.corpus_idr_fraction
python -m extras.figures.pretraining_data.corpus_composition
python -m extras.figures.pretraining_data.composition_vs_disprot
# --- add one line per figure as the other sections are ported in P6 ---
# python -m extras.figures.disorder.disorder_barplot    --kind metapredict --fasta /path/gen.fasta
# python -m extras.figures.biophysics.sparrow_metrics    --metric FCR       --pickles /path/*.pkl
# python -m extras.figures.training_curves.training_curves --logs /path/lightning_logs
# python -m extras.figures.sae.feature_analysis          --features /path/feature_dataset

# --- lab-journal figures (write to journal_figs/ under the lab-journal dir, not the manuscript figs) ---
# Set IDIOM_FIG_DIR to the lab-journal dir and IDIOM_CONDENSATE_RESULTS to the condensate-RL run, e.g.:
#   IDIOM_FIG_DIR=.../IDiom-manuscript-v1/lab_journal \
#   IDIOM_CONDENSATE_RESULTS=.../2026-06-18_rl/04_deeploc \
#   python -m extras.figures.condensate_rl.specificity_matrix
#   python -m extras.figures.condensate_rl.external_validation
#   python -m extras.figures.condensate_rl.naturalness

echo "Done. Now: cd \"\$(dirname \"\$IDIOM_FIG_DIR\")\" && git add figs && git commit -m 'update figs' && git push"
