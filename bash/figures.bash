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

# Output root = the manuscript figs/ dir (override by exporting IDIOM_FIG_DIR yourself).
export IDIOM_FIG_DIR="${IDIOM_FIG_DIR:-/data2/scratch/jxliu2/papers/overleaf/IDiom-manuscript-v1/figs}"
echo "Writing figures to: $IDIOM_FIG_DIR"

# --- add one line per figure as scripts are ported in P6 ---
# python analysis/figures/disorder_barplot.py    --kind metapredict --fasta /path/gen.fasta
# python analysis/figures/disorder_barplot.py    --kind iupred3     --fasta /path/gen.fasta
# python analysis/figures/sparrow_metrics.py      --metric FCR      --pickles /path/*.pkl
# python analysis/figures/training_curves.py      --logs /path/lightning_logs   # validation_loss_vs_steps
# python analysis/figures/rl_curves.py            --logs /path/rl_lightning_logs # mean_entropy, mean_percent_identity

echo "Done. Now: cd \"\$(dirname \"\$IDIOM_FIG_DIR\")\" && git add figs && git commit -m 'update figs' && git push"
