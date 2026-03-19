#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Make prompts for generating IDRs
make_infer_prompt \
    --shard   "${REPO_ROOT}/models/data/shard/0001_file.h5" \
    --out_dir "${REPO_ROOT}/models/data/prompts" \
    specific \
    --fasta        ./example_sequences.fasta \
    --num_duplicates 10000
