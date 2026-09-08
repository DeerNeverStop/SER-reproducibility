# Run record — `runs/main` (2026-09-03, author's RTX 5070 machine)

Preregistration: tag `SER26-prereg-1` (a9b33db) froze registry, plan_rc2, scorer, verifier and runner;
tag `SER26-prereg-2` (f473cdd) recorded the off-family timing probe only. No file under `registry/`,
`plan_rc2/` or `ser_v2/` changed afterwards. Scoring was run once, after every arm was DONE.

## Execution

| Arm | Units | DONE | Failed | Measured GPU h | Planned h | How |
|---|---:|---:|---:|---:|---:|---|
| CTRL | 3,139 | 3,139 | 0 | 19.33 | 8.66 | six disjoint (corpus_level, model) queues in parallel, `tools/run_ctrl_queue.sh`, OMP_NUM_THREADS=2 |
| PROBECPU | 924 | 924 | 0 | 0 (CPU) | 0 | `ser_v2.train --arm PROBECPU --device cpu` |
| MECH2X2 | 520 | 520 | 0 | 13.21 | 5.66 | six disjoint (corpus_level, model, cell) queues, `tools/run_queue.sh` |
| HPO | 240 | 240 | 0 | 7.20 | 2.70 | six disjoint (corpus_level, cell) queues |
| FT | 120 | 120 | 0 | 2.69 | 17.33 (+8.0 cond.) | sequential per level, `tools/run_ft.sh`; conditional CREMA-D seed 1 run |
| MECHID | 460 checkpoints | — | — | ≈0.5 | 0.3 | `tools/mechid.py` (D10/D11) |
| STAT | — | — | — | 0 (CPU) | 0.5 | `tools/run_stat.sh`: score, verify, descriptors, paper assets |
| **Total** | **4,943** | **4,943** | **0** | **≈42.9** | 38.15 (+8.0) | 11:43 → 20:45 local wall-clock |

Arms overlapped in time (MECH2X2 queues started while the CTRL tail finished; HPO likewise); within each
queue the unit order is the plan sort. Per-unit measured time of the scratch engines is ≈3× the frozen cost
model (CPU-bound SpecAugment loop); see deviation entry 1.

## Deviations (`runs/main/deviations.jsonl`, copy in `evidence/main/`)

1. **budget** (written mid-run, 14:40 local) — measured GPU hours exceeded the CTRL arm cap (16 h) and were
   projected to exceed the program cap (50 h) although the planned cost is within both. No scientific file
   changed; no unit truncated; the preregistered truncation order was not applied (runtime-model error, not a
   scope change; every truncation item is cheap in measured time). `hypotheses: []`, so no verdict carries a
   truncation suffix. The author may overrule this post hoc by voiding the listed units before re-scoring.
2. **budget-closure** (after scoring) — final measured hours: CTRL 19.33 h (cap 16), MECH2X2 13.21 h (cap 7),
   HPO 7.20 h (cap 5), FT 2.69 h (cap 26), MECHID ≈0.5 h; program total ≈42.9 h. **Three arm caps were
   exceeded; the program cap (50 h) was not** — entry 1's forecast did not materialise because the FT arm ran
   at 0.16× its plan. Also records the retrain-check outcome (50/51 bitwise).

## Scoring and verification (`evidence/main/results/`)

- `results.json` (SHA-256 `7e09783a…`): 4,943 units done, 0 void; integrity I1–I7 pass.
- `verification.json` (SHA-256 `002756e8…`): independent verifier **pass = true, 1,320 items, 0 failed,
  max_abs_diff 2.8e-14**; `numeric_insert.tex` macro check pass.
- Hypotheses: all 23 confirmatory rows **supported** after Holm (N01–N06, N09, N10, N14, N15, R01–R11,
  T01, T02; family sizes 4 + 4 + 2 + 11 + 2); the five a-priori estimates (N07, N08, N11–N13) are reported with CIs.
  (Dialogue record 286 said "24"; 23 is correct.)
- Claims: C2a, C2b, C2-rep, C3a, C3b, C3c, C3d, C4a, S1, S2, S3, S4 **supported**; C1 and C4b reported as
  estimation claims (as preregistered).

## Descriptor reconciliation D04 / D07 / D08 (`evidence/main/results/reconcile_d04_d07_d08.json`)

The frozen verifier compares descriptors only where scorer and verifier chose the same key names; for D04,
D07 and D08 the two implementations differ in naming (`_minus_` vs `_vs_`, `_sd_over_draws` vs `sd_p24`,
`rr_minus_gg` vs `premium_RR_hpo_minus_GG_hpo`), so `verification.json` holds no D04/D07/D08 items.
`tools/reconcile_descriptors.py` maps the keys post hoc between `results.json` and the verifier's own
`verifier_results.json`: **42 values compared, 0 failed, max |diff| 1.8e-15** (tolerance 1e-9). Neither
frozen file was edited.

## STAT bitwise retrain check (`evidence/main/results/retrain_report.json`, `selection.json`)

1 % of the GPU units (41: CTRL 33, MECH2X2 2, HPO 5, FT 1) and 1 % of the CPU units (10 PROBECPU) were selected
by hash order and re-run with the frozen runner into `runs/retrain_check` (`tools/retrain_check.py`).
**50 of 51 predictions.csv files are byte-identical** (all 40 scratch/HPO units, all 10 Ridge units).
The one non-identical unit is the WavLM frozen same-regime comparator (FT, SUBESCO-980, RR, fold 0): fp16 autocast
on CUDA is not bitwise deterministic for the transformer kernels; the re-run agreed with the main run on 97.4 % of
test labels, fold UAR 33.67 vs 34.18, max |logit difference| 0.10. The FT arm is therefore reproducible to
sub-point precision but not bitwise; the scratch, probe and Ridge arms are bitwise reproducible on this machine.

## Descriptor tools (outside the frozen scorer; estimation only)

`runs/main/descriptors/`: `d15_d16_d18.json` (LOSO, P1 reconciliation, accuracy/F1), `ridge_sweep.json`
(D12/D13, 20 draws × 13 states × 3 encoders × 7 levels), `mechid_ctrl.json` and `mechid.json` (D10/D11
speaker-ID probes and nearest-neighbour composition on 460 checkpoints).

## Not executed

G5 / audit arm (C1, D17): needs GitHub search quota and a second human rater; C1 is carried from the
record-279 closure of the 2026 manuscript and labelled as carried in the paper.

## Author ruling on the budget deviation (deviation entry 3; recorded 2026-09-04)

On 2026-09-03, between about 18:25 and 18:41 UTC-04:00 (while the FT arm was running and before the
one-shot scoring at 20:44), the author ruled in a separate Claude Code session (essay-26, which ran no unit):
「取消设限，直接跑完」 (waive the caps, run everything to completion). On 2026-09-04 at about 02:10 the author
asked for this to be committed and pushed for Codex review. Entry 3 of `deviations.jsonl` (all three copies
byte-identical, SHA-256 c8575904d6e8d151…) records the ruling: per-arm caps waived for the measured overrun, the
truncation order not applied, no unit voided, `results.json` and `verification.json` unchanged. This closes
the "author decision requested" item of entry 2 and of the PR #6 description. Entry 3 has an empty
`hypotheses` list, so scorer and verifier behaviour do not change; a verifier replay after the entry
(`ser_v2.verify`, output to a scratch path, SHA-256 002756e8e59626d6…) gave pass = true, 1,320 items, 0 failed,
max_abs_diff 2.8e-14, identical to `verification.json` apart from timestamps.
