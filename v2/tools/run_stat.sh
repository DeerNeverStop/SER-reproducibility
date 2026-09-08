#!/usr/bin/env bash
# STAT stage (run once, after every arm is DONE): one-shot scoring, independent verifier replay,
# descriptor tools, manuscript assets. Nothing here fits a model.
set -u
cd "$(dirname "$0")/.."
P="E:/科研/SER/ser_gpu/Scripts/python.exe"
LOG=runs/main/logs; mkdir -p "$LOG" runs/main/descriptors ../paper/v2
echo "== STAT start $(date)" >> "$LOG/driver.log"
"$P" -m ser_v2.score  --plan plan_rc2 --manifests manifests --run runs/main --registry registry > "$LOG/score.log" 2>&1; echo "score exit $?" >> "$LOG/driver.log"
"$P" -m ser_v2.verify --plan plan_rc2 --manifests manifests --run runs/main --registry registry --results runs/main/results.json > "$LOG/verify.log" 2>&1; echo "verify exit $?" >> "$LOG/driver.log"
"$P" -m tools.summarize_descriptors --plan plan_rc2 --manifests manifests --run runs/main --out runs/main/descriptors/d15_d16_d18.json > "$LOG/d15_d16_d18.log" 2>&1; echo "d15 exit $?" >> "$LOG/driver.log"
"$P" -m tools.tex_macros --in runs/main/numeric_insert.tex --out ../paper/v2/numeric_insert_paper.tex >> "$LOG/paper_assets.log" 2>&1
"$P" -m tools.paper_tables --plan plan_rc2 --registry registry --run runs/main --out ../paper/v2 >> "$LOG/paper_assets.log" 2>&1
"$P" -m tools.paper_figure --run runs/main --out ../paper/v2/figure_v2.pdf >> "$LOG/paper_assets.log" 2>&1; echo "paper assets exit $?" >> "$LOG/driver.log"
echo "== STAT end $(date)" >> "$LOG/driver.log"
