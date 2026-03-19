#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

make_rl_dataset idr \
    --fasta "$REPO_ROOT/entrypoints/train/post-train/rl_sequence.fasta" \
    --shard "$REPO_ROOT/models/data/shard/0001_file.h5" \
    --out_dir "$REPO_ROOT/models/data/rl_datasets"
