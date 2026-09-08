#!/usr/bin/env bash
# G4 (WavLM engines) + off-family timing probe: one partial-fine-tune unit and one frozen same-regime unit
# of the CREMA-D FT cell (RR, fold 0, seed 0) on SYNTHETIC audio at CREMA-D cardinality and durations.
# Outcome-blind by construction; only the wall-clock per fold feeds tag-2 (FT cap).
set -u
cd "$(dirname "$0")/.."
P="E:/科研/SER/ser_gpu/Scripts/python.exe"
AUD="E:/claudework_data/ICASSP2027-corpora/synthetic_cremad_timing_probe"
LOG=runs/prep/logs; mkdir -p "$LOG"
echo "timing probe start $(date)" > "$LOG/timing_probe.log"
for M in wavlm_base_plus_ft wavlm_base_plus_frozen_sr; do
  "$P" -m ser_v2.train --plan plan_rc2 --manifests manifests --features features --run runs/prep/timing_probe --arm FT \
       --filter corpus_level=cremad,model=$M,cell=RR,fold=0,seed_index=0 --max-units 1 --device cuda --audio-root "$AUD" >> "$LOG/timing_probe.log" 2>&1
  echo "$M done $(date)" >> "$LOG/timing_probe.log"
done
echo "timing probe end $(date)" >> "$LOG/timing_probe.log"
