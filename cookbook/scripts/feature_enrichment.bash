#!/bin/bash
set -euo pipefail

# Run in the environment where you pip-installed IDiom.
REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/enrichment"  # EDIT: new or empty directory
cd "$REPO"

SAE=jxliu2/idiomsae-300M-L18-k32
POSITIVE=cookbook/example_data/protgps/nucleolus.fasta  # EDIT

# Omitting --background downloads the held-out validation FASTA from jxliu2/idiom-data.
# To use your own background, add: --background /path/to/background.fasta
# All valid positives are used; add --max-positive 128 for a smaller run.
idiom_feature_enrichment \
    --sae "$SAE" \
    --positive "$POSITIVE" \
    --out "$OUT" \
    --name nucleolus \
    --top-n 30 \
    --max-background 10000 \
    --batch-size 4 \
    --seed 0
