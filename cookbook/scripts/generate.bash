#!/bin/bash

set -euo pipefail

###
# Generate de novo IDRs and write them to a FASTA.
# A GPU is recommended; reduce BATCH_SIZE to use less memory.
###

# Run in the environment where you pip-installed IDiom; the clone supplies cookbook files.
REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/generate"  # EDIT: output directory

cd "$REPO"
mkdir -p "$OUT"

MODEL=jxliu2/idiom-300M                 # EDIT: Hub ID, released directory, or training checkpoint
N=100                                 # EDIT: total number of IDRs
BATCH_SIZE=8                          # EDIT: sequences per model forward

idiom_generate unprompted \
    --model "$MODEL" \
    --out "$OUT/idrs.fasta" \
    --n "$N" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens 1000 \
    --temperature 1.0 \
    --seed 0

echo "DONE -> $OUT/idrs.fasta"
