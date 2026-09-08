# Study II execution contract (2026-09-05)

Scope: prospectively specified estimation of fixed data policies; no configuration
search in the core. No new outcome is viewed until the full core is closed.
All JSON hashes use UTF-8, ensure_ascii=False, sort_keys=True, separators=(',',':').
`plan_sha256` hashes all top-level fields except itself. `unit_id` is the full
SHA256 of all unit fields except itself. LF output. Paths are relative to repo
or to the explicitly supplied feature directory; no silent cache discovery.

## Core plan

Schema `ser-study2-core-1`, fields:
- `program`: `SER26-STUDY2-CORE-1`.
- `input`: `manifest_path`, `manifest_sha256`, `feature_sha256` mapping kind to
  hash, `feature_files` mapping kind to explicit basename. Kinds: `logmel`,
  `wavlm_base_plus`.
- `source_sha256`: repo-relative runtime source files and expected byte hashes.
- `design`: protocol constants, representative rule, seeds, estimation status.
- `units`: 1440 unit dictionaries; 720 each for `ridge_wavlm` / `cnn`.
- `plan_sha256`.

Unit fields: `model`, `fold` (0..4), `rotation` (0..5), `draw` (0..2),
`scenario` (`prompt_seen`/`prompt_new`), `B` (288/576), `S` (12/48),
`P_global` (8), `P_per_speaker` (B/(6*S)), `R` (1), `train_seed`,
`fit`, `val`, `test` (sorted relative audio paths), `config`, `unit_id`.

`config` Ridge: `model= ridge_a1_wavlm_base_plus`, `feature_state=12`,
`alpha=1.0`, `class_weight=balanced`, `solver=lsqr`, `tol=0.0001`.
CNN: `model=cnn`, `batch_size=32`, `epochs=100`, `patience=15`, plus explicitly
recorded `lr`, `weight_decay`, `dropout` from existing candidate index 2.
The runtime code SHA also pins inherited architecture/augmentation defaults.

Ridge uses the state-12 cache with train-only StandardScaler and explicit solver.
CNN read-only adapts engine_p1_frozen using strict manifest/cache wrappers, never
PlanIO fallback or FeatureStore's ambiguous glob. All labels come from the pinned
real manifest; hygienic byte duplicates must not cross boundaries. Fit, val,
test speakers disjoint; class index set 0..5. Test is same within each
draw/fold/rotation across every policy, scenario, model and budget. Val is the
same 8 stop speakers on the common six prompts in both scenarios. Ridge ignores
val but shares fitpool with CNN. No S_select in core.

## Runner output and closure

CLI: `python -m v3.data_design.core_run --repo REPO --plan FILE --features DIR
--out DIR --model ridge_wavlm --device cpu [--max-units N] [--threads 2]`.
CNN requires `--model cnn --device cuda --gpu-release FILE`. That receipt must
be schema `ser-study2-gpu-release-1`, `released=true`, `n14r_verification_pass=true`,
and identify SHA256-checked local `completion_path`, `analysis_lock_path`, and
`verification_path` with corresponding `_sha256` fields. This is an explicit
handoff created after independent N14R validation, not an automatically assumed
date. No cloud/GPU provisioning in this CLI.

An independent cloud host may instead use schema
`ser-study2-independent-host-1`: `authorized=true`, `study2_only=true`,
`host` matching the execution hostname, `local_n14r_host` different from it,
`plan_sha256` matching this core, and an unexpired UTC `expires_at`. This receipt
is written only after the user authorizes a concrete rental and configuration;
it does not buy a server. An independent host does not wait for local N14R.

Each completed unit directory is `units/<unit_id>/`, with `predictions.npz`
(`paths`,`logits` for test; no performance), `unit.json`, and `DONE` JSON.
Receipt includes unit_id, plan_sha256, actual environment, fit/wall time,
prediction sha256; DONE binds byte SHA256 of both files. Resume validates exact
identity, file hashes, finite prediction dimensions and path order. Different
plan, corrupted output or active lock must be rejected. An interrupted partial
unit may be retried only with an explicit event; no result becomes DONE early.
Each unit has at most two fit attempts including orphaned starts; exhaustion
leaves the core incomplete and forbids automatic third tries or seed changes.
Concurrent runners writing one output directory are prohibited.

Runner prints only operational progress. Once all 1440 unique units verify,
write `completion.json` and `analysis_lock.json` binding plan and all DONE hashes.
`--max-units` / one-model completion never locks a partial core. Score reads only
the complete locked plan. Model/runtime timing may be inspected beforehand;
scientific accuracy, confidence intervals, ranks and effect sizes may not.

`core_plan.py` is a standalone builder; runner should independently validate
JSON/input/source hashes and invariants, without trusting its implementation.
Independent verifier will be implemented separately. A formal public time stamp
is not implied by this document: code and metadata identity are committed and
pushed before core training; prior data exposure is disclosed separately.
