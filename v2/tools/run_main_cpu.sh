#!/usr/bin/env bash
# CPU arm (PROBECPU: Ridge contract units) in parallel with the GPU driver; then the descriptor sweep (D12/D13).
set -u
cd "$(dirname "$0")/.."
P="E:/科研/SER/ser_gpu/Scripts/python.exe"
LOG=runs/main/logs; mkdir -p "$LOG" runs/main/descriptors
echo "CPU driver start $(date)" >> "$LOG/driver_cpu.log"
"$P" -m ser_v2.train --plan plan_rc2 --manifests manifests --features features --run runs/main --arm PROBECPU --device cpu >> "$LOG/train_PROBECPU.log" 2>&1
echo "== PROBECPU exit $? $(date)" >> "$LOG/driver_cpu.log"
"$P" -m tools.ridge_sweep --plan plan_rc2 --manifests manifests --features features --out runs/main/descriptors/ridge_sweep.json >> "$LOG/ridge_sweep.log" 2>&1
echo "== ridge_sweep exit $? $(date)" >> "$LOG/driver_cpu.log"
echo "CPU driver done $(date)" >> "$LOG/driver_cpu.log"
