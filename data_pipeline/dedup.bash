#!/bin/bash
# Stage 3 — DisProt leakage dedup (operator; mmseqs heavy). Removes record-FASTA IDRs that are
# >=50% identical to any DisProt IDR (ESM-2 dedup recipe), IDR-vs-IDR. Run via Slurm:
#   srun -c 64 --mem 64GB -t 12:00:00 bash data_pipeline/dedup.bash <records.fasta> <work_dir>
set -euo pipefail

RECORD_FASTA=${1:?usage: dedup.bash <records.fasta> <work_dir> [disprot.json]}
WORK=${2:?usage: dedup.bash <records.fasta> <work_dir> [disprot.json]}
DISPROT_JSON=${3:-/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data/reference/disprot/"DisProt release_2025_06 with_ambiguous_evidences.json"}
THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$WORK"

# 1. Build query (DisProt IDRs) and target (record-FASTA IDR substrings).
python -m data_pipeline.dedup disprot-fasta --json "$DISPROT_JSON" --out "$WORK/disprot_idrs.fasta"
python -m data_pipeline.dedup idr-fasta     --fasta "$RECORD_FASTA" --out "$WORK/train_idrs.fasta"

# 2. mmseqs search: DisProt (query) -> train IDRs (target) at >=50% id, 80% cov (ESM-2 params).
mmseqs createdb "$WORK/disprot_idrs.fasta" "$WORK/qDB" -v 1
mmseqs createdb "$WORK/train_idrs.fasta"   "$WORK/tDB" -v 1
mmseqs search  "$WORK/qDB" "$WORK/tDB" "$WORK/res" "$WORK/tmp" \
  --min-seq-id 0.5 -c 0.8 --cov-mode 0 -s 7 --max-seqs 300 --alignment-mode 3 --threads "$THREADS" -v 3
mmseqs createtsv "$WORK/qDB" "$WORK/tDB" "$WORK/res" "$WORK/hits.tsv" -v 1

# 3. Matched train headers (col 2 = target) -> drop them from the full-seq record FASTA.
cut -f2 "$WORK/hits.tsv" | sort -u > "$WORK/remove.txt"
python -m data_pipeline.dedup remove --fasta "$RECORD_FASTA" --remove "$WORK/remove.txt" \
  --out "$WORK/train_dedup.fasta"

echo "done -> $WORK/train_dedup.fasta  (removed $(wc -l < "$WORK/remove.txt") records)"
