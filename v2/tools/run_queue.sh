#!/usr/bin/env bash
# Run one disjoint queue of an arm: bash tools/run_queue.sh <ARM> <queue index> [device]
# Reads runs/main/<ARM>_queue_<i>.txt (one --filter string per line); resumable via DONE markers.
set -u
cd "$(dirname "$0")/.."
P="E:/科研/SER/ser_gpu/Scripts/python.exe"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
ARM="$1"; Q="$2"; DEV="${3:-cuda}"; LOG=runs/main/logs; mkdir -p "$LOG"
EXTRA="${AUDIO_ROOT_ARGS:-}"
echo "== $ARM queue $Q start $(date)" >> "$LOG/driver.log"
while IFS= read -r F; do
  [ -z "$F" ] && continue
  "$P" -m ser_v2.train --plan plan_rc2 --manifests manifests --features features --run runs/main --arm "$ARM" --device "$DEV" --filter "$F" $EXTRA >> "$LOG/train_${ARM}_q$Q.log" 2>&1
done < "runs/main/${ARM}_queue_$Q.txt"
echo "== $ARM queue $Q end $(date)" >> "$LOG/driver.log"
