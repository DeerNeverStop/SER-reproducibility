# N14R execution and scoring contract

`spec.json`, `spec.py`, and this document are frozen together. Any disagreement is fatal and requires a newly versioned preregistration made before further execution.

## Plan and split schema

`plan/run_plan.csv` has the exact ordered columns exposed by `spec.PLAN_FIELDS`. There are 2,240 unique planned rows: 80 per draw. A primary execution initially admits draws 0--23 only. Reserve draws 24--27 become eligible one at a time, in order, solely after an earlier draw is void.

`plan/hygiene.json` commits the deterministic H1/H2/H3 transformation from the 7,442-row byte-pinned manifest to the 7,435-row analysis population, including all excluded paths and reasons. There are 56 old-`PlanIO`-compatible split files, one per draw and cell: `plan/splits/n14r__dDD__CELL.json`. Each contains that exact cleaned `population` (relative paths) and five `{fold,test,val,fit,meta}` entries. `plan/split_index.json` maps each key to `{sha256,path,n_folds}`; a plan row's `split_sha256` is the SHA of its complete cell split file. `GR_hpo` and `GG_hpo` have different inner partitions/files but their test arrays and fold-level `outer_sha256` values are exactly equal. Every fit/validation/test partition contains all six classes. `GR_hpo` has nonzero inner speaker overlap, `GG_hpo` has zero, and `inner_overlap_speaker_count/fraction` are recorded in fold metadata. `plan/unit_configs.json` maps each `config_sha256` to a complete old-engine-compatible configuration.

## Unit output schema

For plan row `unit_id=U`, execution atomically writes, in order:

1. `runs/n14r/units/U/predictions.csv`, the existing trainer's outer-test prediction table with `sample_index,relative_path,speaker,y_true,y_pred,logit_0..logit_5`.
2. `runs/n14r/units/U/history.json`, containing every completed epoch and at least `epoch,val_loss,val_uar`.
3. `runs/n14r/units/U/unit.json`, which repeats the old-plan identity/hash fields, contains the full `config`, `predictions_sha256`, `val_uar_best`, and `best_epoch`.
4. `runs/n14r/units/U/DONE`, the existing trainer's prediction SHA-256 followed by LF.

A mismatched, missing, duplicate, extra, non-finite, or out-of-range prediction invalidates the unit. The scorer joins test predictions to the frozen split and recomputes test macro recall with all labels `[0,1,2,3,4,5]`, multiplied by 100. It verifies `val_uar_best` against the `val_uar` on the history row having the strict minimum `val_loss`, and verifies `best_epoch` against that row. A tie in validation loss is resolved by the first epoch, matching the strict-improvement training loop.

## Selection and scoring

Within each draw/fold/cell, choose the configuration having greatest `val_uar_best` after verifying that value and `best_epoch` against the first strict minimum-`val_loss` row in `history.json`; exact UAR ties choose the smallest integer index. The selected configuration's test UAR is recomputed from `predictions.csv`. Compute N14R only by the fold-then-draw formulas in the preregistration. Speakers, samples, folds, and fits must never be supplied as independent observations to the inferential test.

An outcome-blind completion gate writes `analysis_lock.json` before any scoring. Its frozen payload contains `schema,study_id,created_at,contains_outcome_statistics,complete_draw_ids,n_complete_draws,n_locked_units,environment_fingerprint,plan_hashes,preregistration_hashes,external_feature_cache,execution_ledger_sha256,audit_ledger_prefix_sha256,unit_artifact_hashes,lock_payload_sha256`. `plan_hashes` covers `run_plan.csv`, `split_index.json`, `unit_configs.json`, `plan_summary.json`, and `hygiene.json`; `preregistration_hashes` covers `spec.json` and `PINS.json`; `external_feature_cache` records the exact CREMA-D log-mel cache and metadata path, size, and SHA-256 verified by the runner before fitting; `unit_artifact_hashes` maps every locked unit ID to exact-byte SHA-256 values for its four files. The runner parses individual histories/predictions before declaring a unit complete only to check schema, finiteness, best-checkpoint consistency, hashes, and exact split/manifest coverage; it performs no HPO selection or fold/cell/draw aggregation. The environment fingerprint and both ledger hashes are those written by the frozen runner. Only then may the scorer produce results; afterward the independent verifier replays scoring and compares outputs. Both fail closed if the pilot appears, a draw has fewer or more than 80 units, a grid is incomplete, cells differ in outer hash/test rows/train seed, or reserve order is violated.

The primary output contains `n_draws=24`, draw IDs, all draw deltas, mean, SD, SE, t, df, raw two-sided p, Holm p, 95% t-CI, and the directional verdict. Sensitivity outputs contain the frozen bootstrap and full sign-flip results and are explicitly non-primary. D09R contains only draw values, mean, SD, and 95% t-CI.

## Retry and ledger events

The inherited trainer appends immutable per-unit start, completion, and failure events; the N14R wrapper appends draw activation, pass completion, draw completion/void, and lock events. The count of a unit's failure events defines its allowance: one initial attempt plus at most one identical retry. A retry must reproduce the same frozen plan row exactly. After an interrupted process, an unmatched `start` is closed as `done` only if all four atomic artifacts pass the full preflight; otherwise it is recorded as a failed attempt. At the next wrapper gate, two recorded failures permanently void the draw even if a later, externally produced `DONE` artifact exists; other outputs completed in the same trainer pass are excluded with that draw. Reserve activation is a separate wrapper-ledger event. Across 2,240 possible unique units the theoretical ceiling is 4,480 attempts. The 125 GPU-hour allocation is soft and unenforced; the fixed attempt/reserve rule is the stopping rule. A final `analysis_lock` event points to the lock file covering the 24 included complete draw IDs, plan, unit artifacts, and ledger prefix. No outcome statistic is a permitted ledger field before `analysis_lock`.
