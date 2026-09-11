#!/bin/bash

set -euo pipefail

# Build a per-residue SAE feature dataset from a record FASTA
# The SAE release supplies its host model, layer, and prompt format; the released SAE encodes IDRs
# A GPU is recommended; reduce BATCH_SIZE to use less memory

REPO="/path/to/idiom" # EDIT: repository checkout
OUT="/path/to/output/features" # EDIT: feature dataset directory

cd "$REPO"

SAE=jxliu2/idiomsae-300M-L18-k32 # EDIT: Hub ID or released SAE directory
FASTA=cookbook/example_data/protgps/nucleolus.fasta # EDIT: headers end in _IDR_x-y
BATCH_SIZE=16 # EDIT: records per model forward

idiom_feature_dataset \
    --sae "$SAE" \
    --fasta "$FASTA" \
    --out "$OUT" \
    --batch-size "$BATCH_SIZE"

# Inspect the output with FeatureDataset in Python or launch the viewer from this checkout:
# streamlit run src/idiom/sae/features/feature_viewer.py -- --features "$OUT"
echo "DONE -> $OUT"
