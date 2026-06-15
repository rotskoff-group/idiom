#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --time=7-00:00:00
#
# Curation stage 4b (optional) — SignalP signal-peptide filter. Trims/drops record IDRs that
# overlap an N-terminal signal peptide (SignalP 6 fast, GPU-converted). SignalP is
# CPU-preprocessing-bound (~450 seq/s/process on GPU), so we run W = NUM_GPUS * WORKERS_PER_GPU
# worker processes round-robin across the **Slurm-allocated** GPUs to use CPU + GPU well. The GPU
# count comes from the allocation (--gres below), NOT an env var; workers are pinned to the
# allocated physical ids (this cluster does not isolate GPUs). Submit:
#
#   sbatch data_pipeline/signalp_filter.bash                         # uses the 4 GPUs in --gres
#   WORKERS_PER_GPU=6 MODE=trim MIN_PROB=0.9 sbatch ...              # more workers per GPU
#   sbatch --gres=gpu:2 data_pipeline/signalp_filter.bash           # change GPU count via --gres/--gpus
#
# One-time prereq (GPU model conversion, already done at $GPUMODELS):
#   signalp6_convert_models gpu <models_dir>     # cpu->cuda; see signalp/conversion_utils
set -euo pipefail

# --- tunables ---
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
THREADS_PER_WORKER=${THREADS_PER_WORKER:-4}
BSIZE=${BSIZE:-256}
MODE=${MODE:-trim}             # trim | drop
MIN_PROB=${MIN_PROB:-0.9}      # min predicted-class probability to act on a signal call
MIN_LEN=${MIN_LEN:-30}         # trim: keep mature remainder only if >= this

# GPUs ACTUALLY allocated by Slurm. CUDA_VISIBLE_DEVICES holds the physical ids (e.g. "4,5,6,7");
# this cluster does NOT cgroup-isolate GPUs, so workers must be pinned to THESE ids — never absolute
# 0..N-1, or we'd land on another job's GPUs. NUM_GPUS is derived from the allocation, not hardcoded.
IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated (run under sbatch/srun with --gres=gpu:N)}"
NUM_GPUS=${#GPU_IDS[@]}
W=$((NUM_GPUS * WORKERS_PER_GPU))

# --- envs / tools ---
REPO=/data2/scratch/jxliu2/idiom
source "$REPO/.venv/bin/activate"          # idiom env: the pure python helpers
cd "$REPO"
SP_VENV=/data2/scratch/jxliu2/tmp/deeptmhmm-venv          # has signalp6 + GPU torch (cu121)
SIGNALP="$SP_VENV/bin/signalp6"                           # console script -> its own interpreter
GPUMODELS=/data2/scratch/jxliu2/tmp/signalp_gpu_models    # GPU-converted distilled model

# --- paths ---
DATA=/data2/scratch/group_scratch/idr_plm/2026-06-14_idiom_data
AFDB="$DATA/pretraining/AFDB"
RECORD_FASTA="${RECORD_FASTA:-$AFDB/intermediate/AFDB_IDR_90_len1020_rm_full_low_plddt_dedup_disprot.fasta}"
WORK="${WORK:-$AFDB/intermediate/signalp}"
PROT="$WORK/proteins.fasta"
OUT_FASTA="${OUT_FASTA:-$AFDB/intermediate/$(basename "${RECORD_FASTA%.fasta}")_signalp.fasta}"
mkdir -p "$WORK/shards" "$WORK/results"

[ -x "$SIGNALP" ]            || { echo "signalp6 not found at $SIGNALP"; exit 1; }
[ -f "$GPUMODELS/distilled_model_signalp6.pt" ] || { echo "GPU model missing at $GPUMODELS (run signalp6_convert_models gpu)"; exit 1; }
echo "records: $RECORD_FASTA"
echo "workers: $W ($NUM_GPUS gpus x $WORKERS_PER_GPU)  threads/worker=$THREADS_PER_WORKER  mode=$MODE min_prob=$MIN_PROB"
echo "out    : $OUT_FASTA"

# 1. unique parent-protein FASTA (SignalP input) + shard W ways.
[ -f "$PROT" ] || python -m data_pipeline.signalp_filter protein-fasta --fasta "$RECORD_FASTA" --out "$PROT"
python -m data_pipeline.signalp_filter shard --fasta "$PROT" --n "$W" --out-prefix "$WORK/shards/shard"

# 2. launch W SignalP workers, round-robin across GPUs.
pids=()
for ((w = 0; w < W; w++)); do
  gpu=${GPU_IDS[$((w % NUM_GPUS))]}        # pin to an allocated physical GPU id
  rm -rf "$WORK/results/r_$w"
  CUDA_VISIBLE_DEVICES="$gpu" "$SIGNALP" \
    --fastafile "$WORK/shards/shard_$w.fasta" --organism other \
    --output_dir "$WORK/results/r_$w" --format none --mode fast \
    --model_dir "$GPUMODELS" --bsize "$BSIZE" --torch_num_threads "$THREADS_PER_WORKER" --write_procs 1 \
    > "$WORK/results/r_$w.log" 2>&1 &
  pids+=($!)
done
echo "launched $W workers; waiting ..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a SignalP worker failed; see $WORK/results/*.log"; exit 1; }

# 3. aggregate all shard predictions -> confident signal calls -> trim/drop the record FASTA.
python -m data_pipeline.signalp_filter apply \
  --record-fasta "$RECORD_FASTA" \
  --results "$WORK"/results/r_*/prediction_results.txt \
  --mode "$MODE" --min-prob "$MIN_PROB" --min-len "$MIN_LEN" \
  --out "$OUT_FASTA" --remove-list "$WORK/removed.txt"

echo "done -> $OUT_FASTA"
echo "scratch (rm -rf when done): $WORK/shards $WORK/results"
echo "Next: stage 5 random split -> python -m data_pipeline.split --fasta \"$OUT_FASTA\" --out-dir \"$AFDB/splits\""
