#!/usr/bin/env bash
# FT arm (WavLM-base+ partial fine-tune vs frozen same-regime comparator), sequential: one process holds ~10 GB.
# Each family level has its own audio root; unit order within a level follows the plan sort. Resumable.
set -u
cd "$(dirname "$0")/.."
P="E:/科研/SER/ser_gpu/Scripts/python.exe"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
LOG=runs/main/logs; mkdir -p "$LOG"
echo "== FT start $(date)" >> "$LOG/driver.log"
for spec in "ravdess|E:/科研/SER/data" "subesco_980|E:/claudework_data/ICASSP2027-corpora/subesco/extracted" "cremad|E:/科研/SER/AudioWAV"; do
  IFS='|' read L R <<< "$spec"
  "$P" -m ser_v2.train --plan plan_rc2 --manifests manifests --features features --run runs/main --arm FT --device cuda --filter "corpus_level=$L" --audio-root "$R" >> "$LOG/train_FT.log" 2>&1
  echo "== FT $L exit $? $(date)" >> "$LOG/driver.log"
done
echo "== FT end $(date)" >> "$LOG/driver.log"
