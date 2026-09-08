# N14R2 historical scientific snapshot / 历史科学快照

Source commit: `4fe0bb52f15b56a6626363be31a378b0f9661293`, exported from the already-fetched `origin/codex/n14r2-cloud-20260905` of `DeerNeverStop/SER`.
Original release: https://github.com/DeerNeverStop/SER/releases/tag/SER26-N14R2-complete-20260906 . That provenance repository remains private; this directory makes the selected scientific materials independently accessible. Publication of this snapshot today does not establish public preregistration at the historical execution date.

## What this contains

Original-byte N14R2 scientific specification, source, selected original tests, package requirements, and generated plans (24 primary draws plus four reserves). The completed study used 24 draws × 5 folds × 2 inner-validation conditions × 8 configurations = 1,920 new fits. Reserve plans are preserved as plans; their presence does not mean they were fitted.

`dependencies/` preserves the explicitly pinned supporting Python sources and CREMA-D manifest at the historical revision, retaining their original repository-relative layout beneath that directory. These are source records, not an automatically installed replacement for the public repository's current modules. CREMA-D metadata keeps the upstream database/content rights; this snapshot does not relicense third-party data.

`results/` preserves the original result, original failed independent verification, corrected result and corrected verification, and the explicit metadata erratum. The failed report is intentional evidence, not an unresolved failure silently replaced by a passing file. The correction is documented by `n14r2_metadata-erratum.json` and `metadata_erratum.py`; use the metadata-corrected result for interpretation and retain both histories. Each included release file was checked against its GitHub release asset SHA-256 before copying. `EXPORT_MANIFEST.json` records every source and exported SHA; exported payloads are byte-identical.

## Invalid earlier inference and replacement

The earlier v2 N14 inference expanded fold-level validation values into speaker-level rows and treated the resulting 91 rows as independent, creating pseudoreplication. Its original inferential conclusion is invalid and must not be combined with N14R2 as another valid independent result. N14R2 supplies the replacement draw-level analysis. The intervening N14R run was terminated incomplete; its 1,298 successful fits and 16 complete draws are excluded from N14R2. N14R2 is a fresh study, not a continuation or a repaired old ledger.

## Explicit reproduction limits

This is a scientific source/plan/small-results snapshot, **not a one-command complete retraining or a full archived-run authentication package**. It excludes the 486,594,560-byte `n14r2-closed-run.tar`, per-fit predictions/checkpoints, raw audio, feature cache, cloud launch/control code, live-machine environment receipts/bindings, original PINS and `analysis_lock.json` containing machine identities. The lock's original SHA is recorded as an omitted source; it was not copied or silently redacted. `EXPORT_MANIFEST.json` identifies omitted files and their hashes; those hashes are provenance references, not proof that absent artifacts were reverified here.

The unchanged historical runner calls `.ops.environment`; its original freeze/admission needs the omitted node bindings and PINS. `worker.py` also imports `v2.ser_v2` and `v2_1.n14r.run`; pinned copies are in `dependencies/`. Therefore a direct `python -m v2_1.n14r2.run` from this public snapshot is not advertised as runnable. Restoring the exact original layout, licensed input bytes and machine-bound admission would require the original private archive or a separately documented new execution protocol, not edits disguised as the historical run.

`requirements-lock.txt` is the original Linux/Python 3.12 dependency lock, not a promise that it installs on every platform. Selected tests may require omitted archival fixtures or operational prerequisites. Python files were syntax-parsed only during this export; no historical tests, model training or archive-level verification were rerun. Package result tables and the prior decomposition audit support numerical inspection; they do not replace absent model outputs.

Original scientific-source bytes and prior failure history were not modified. This export performed no cloud actions and contains no `ops/` or `environment/` directory.
