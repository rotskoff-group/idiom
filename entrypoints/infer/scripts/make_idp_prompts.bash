#!/bin/bash

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR" && git rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

# Make prompts for generating IDPs
make_infer_prompt \
    --out_dir "${REPO_ROOT}/models/data/prompts" \
    idp \
    --num_duplicates 100
