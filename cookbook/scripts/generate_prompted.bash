#!/bin/bash

set -euo pipefail

# Replace annotated IDRs using their flanking protein context from the bundled DisProt FASTA
# Headers must end in _IDR_x-y (1-based, inclusive); preserve the full sequences and spans
# A GPU is recommended; reduce BATCH_SIZE to use less memory

REPO="/path/to/idiom"  # EDIT: repository checkout
OUT="/path/to/output/generate-prompted"  # EDIT: output directory

cd "$REPO"
mkdir -p "$OUT"

MODEL=jxliu2/idiom-300M                 # EDIT: Hub ID, released directory, or training checkpoint
FASTA=cookbook/example_data/disprot/disprot_len1020_idrs.fasta  # EDIT
N=10                                  # EDIT: replacements per input protein, not total
BATCH_SIZE=8                          # EDIT: sequences per model forward

# --return-full splices each replacement into its flanks and updates the FASTA IDR span.
# Remove it to write only the generated IDRs.
idiom_generate prompted \
    --model "$MODEL" \
    --fasta "$FASTA" \
    --out "$OUT/redesigned.fasta" \
    --n "$N" \
    --return-full \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens 1000 \
    --temperature 1.0 \
    --seed 0

echo "DONE -> $OUT/redesigned.fasta"
