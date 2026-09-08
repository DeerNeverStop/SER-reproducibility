#!/usr/bin/env bash
# Preregistered GPU order after tag-1: CTRL -> MECH2X2 -> HPO. (PROBECPU runs in parallel on CPU via run_main_cpu.sh;
# FT and MECHID start only after the timing probe / tag-2.)  Resumable: the runner skips DONE units.
set -u
cd "$(dirname "$0")/.."
P="E:/科研/SER/ser_gpu/Scripts/python.exe"
LOG=runs/main/logs; mkdir -p "$LOG"
echo "GPU driver start $(date)" >> "$LOG/driver.log"
for ARM in CTRL MECH2X2 HPO; do
  echo "== $ARM start $(date)" >> "$LOG/driver.log"
  "$P" -m ser_v2.train --plan plan_rc2 --manifests manifests --features features --run runs/main --arm "$ARM" --device cuda >> "$LOG/train_${ARM}.log" 2>&1
  echo "== $ARM exit $? $(date)" >> "$LOG/driver.log"
done
echo "GPU driver done $(date)" >> "$LOG/driver.log"
