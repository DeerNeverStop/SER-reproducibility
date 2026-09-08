#!/usr/bin/env bash
# Run one disjoint CTRL queue (list of --filter strings) sequentially; resumable via DONE markers.
set -u
cd "$(dirname "$0")/.."
P="E:/科研/SER/ser_gpu/Scripts/python.exe"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"   # six queues on 16 cores: cap intra-op threads
Q="$1"; LOG=runs/main/logs; mkdir -p "$LOG"
echo "== CTRL queue $Q start $(date)" >> "$LOG/driver.log"
while IFS= read -r F; do
  [ -z "$F" ] && continue
  "$P" -m ser_v2.train --plan plan_rc2 --manifests manifests --features features --run runs/main --arm CTRL --device cuda --filter "$F" >> "$LOG/train_CTRL_q$Q.log" 2>&1
done < "runs/main/ctrl_queue_$Q.txt"
echo "== CTRL queue $Q end $(date)" >> "$LOG/driver.log"
