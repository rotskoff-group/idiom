#!/bin/bash
#SBATCH --job-name=dedup
#SBATCH --output=%x-%j.out
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#
# Curation stage 4 — DisProt leakage dedup (mmseqs, heavy). Removes record-FASTA IDRs that are
# >=50% identical to any benchmark DisProt IDR (ESM-2 recipe), IDR-vs-IDR, cov-mode 0. Self-
# contained: every path is spelled out below. Submit with:
#
#   sbatch data_pipeline/dedup.bash
set -euo pipefail

REPO=/data2/scratch/jxliu2/idiom
source "$REPO/.venv/bin/activate"
cd "$REPO"

# --- paths (all spelled out; override via env if needed) ---
DATA=/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data
AFDB="$DATA/pretraining/AFDB"
RECORD_FASTA="${RECORD_FASTA:-$AFDB/intermediate/AFDB_IDR_90_len1020_rm_full_low_plddt.fasta}"
OUT_DIR="${OUT_DIR:-$AFDB/intermediate}"
DISPROT_JSON="${DISPROT_JSON:-$DATA/reference/disprot/DisProt release_2025_06 with_ambiguous_evidences.json}"
# Parsed DisProt IDR benchmark (1,665 seqs): derived *reference* artifact, kept with the other
# reference datasets and reused as the eval benchmark set.
DISPROT_IDR_FASTA="${DISPROT_IDR_FASTA:-$DATA/reference/disprot/disprot_idrs_len1020.fasta}"
# Heavy, regenerable mmseqs DBs + the big IDR-target fasta -> a self-contained scratch dir under
# /data2/scratch/jxliu2/tmp so they're easy to find and `rm -rf` later (NOT kept with the output).
SCRATCH="${MMSEQS_SCRATCH:-/data2/scratch/jxliu2/tmp/mmseqs_dedup_disprot}"
# mmseqs static binary (avx2) lives here; put it on PATH for this job.
export PATH="/data2/scratch/jxliu2/tmp/bin:$PATH"
MMSEQS="${MMSEQS:-mmseqs}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT_FASTA="$OUT_DIR/$(basename "${RECORD_FASTA%.fasta}")_dedup_disprot.fasta"
mkdir -p "$OUT_DIR" "$SCRATCH"
echo "records : $RECORD_FASTA"
echo "disprot : $DISPROT_JSON"
echo "out     : $OUT_FASTA"
echo "scratch : $SCRATCH"

# 1. Build query (canonical DisProt IDRs -> reference/) and target (record-FASTA IDR substrings).
python -m data_pipeline.dedup disprot-fasta --json "$DISPROT_JSON" --out "$DISPROT_IDR_FASTA"
python -m data_pipeline.dedup idr-fasta     --fasta "$RECORD_FASTA" --out "$SCRATCH/train_idrs.fasta"

# 2. mmseqs search: DisProt (query) -> train IDRs (target) at >=50% id, 80% cov (ESM-2 params).
"$MMSEQS" createdb "$DISPROT_IDR_FASTA"        "$SCRATCH/qDB" -v 1
"$MMSEQS" createdb "$SCRATCH/train_idrs.fasta" "$SCRATCH/tDB" -v 1
"$MMSEQS" search  "$SCRATCH/qDB" "$SCRATCH/tDB" "$SCRATCH/res" "$SCRATCH/tmp" \
  --min-seq-id 0.5 -c 0.8 --cov-mode 0 -s 7 --max-seqs 300 --alignment-mode 3 --threads "$THREADS" -v 3
"$MMSEQS" createtsv "$SCRATCH/qDB" "$SCRATCH/tDB" "$SCRATCH/res" "$OUT_DIR/hits.tsv" -v 1

# 3. Matched train headers (col 2 = target) -> drop them from the full-seq record FASTA.
cut -f2 "$OUT_DIR/hits.tsv" | sort -u > "$OUT_DIR/remove.txt"
python -m data_pipeline.dedup remove --fasta "$RECORD_FASTA" --remove "$OUT_DIR/remove.txt" \
  --out "$OUT_FASTA"

echo "done -> $OUT_FASTA  (removed $(wc -l < "$OUT_DIR/remove.txt") records)"
echo "scratch (rm -rf when done): $SCRATCH"
echo "Next: stage 5 random split -> python -m data_pipeline.split --fasta \"$OUT_FASTA\" --out-dir \"$AFDB/splits\""
