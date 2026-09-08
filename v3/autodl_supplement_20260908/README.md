# AutoDL supplement implementation (2026-09-08)

**Completion note, 2026-09-08:** all 720 formal fits, eight technical pilots, archive verification, and frozen scoring have completed. All ten prespecified primary tests are retained. Independent numerical replay matched 199,780 scalar values with maximum absolute difference 7.105427357601002e-15; full archive verification covered 6,216 files and 20 retained checkpoints. Read the [Chinese results report](../../docs/autodl-supplement-20260908/RESULTS_REPORT_ZH.md), [original result files](../../docs/autodl-supplement-20260908/results/), and [verification/closeout evidence](../../docs/autodl-supplement-20260908/evidence/). The [current five-page PDF](../../paper/supplement-results-20260908/english/xie.pdf) has passed full-page visual inspection and numerical/claim review.

The instance is shut down, its release request returned Success, and two complete list queries no longer contained it; the last status response was `removing`. Account snapshots and their billing limits are in the [cloud closeout report](../../docs/autodl-supplement-20260908/evidence/CLOUD_CLOSEOUT_REPORT.md).

The [prospective protocol and Claude response](../../docs/autodl-supplement-20260908/README.md) remain preserved as the pre-result record. This isolated program leaves the old completed engine/results unchanged. The implementation and reproduction commands below describe distinct input, GPU qualification, completion, and scoring gates; CPU tests or plan generation alone never establish formal completion.

## Portable inputs

- `plans/reference.json.gz`: 360 original A panels and 8,308 required audio metadata rows, without audio, credentials or machine paths.
- `plans/PLAN.json.gz`: 720 formal units, 8 technical pilots, exact role slots, stochastic seeds, recovery sample and ten-test family.
- `plans/MODELS.json`: base weight byte/semantic identities; a deployment-local copy adds model paths.
- Reference semantic SHA: `0cf8b3e0cd72975516540977f4835fcd185ec66ffdb77c87979c661e9f805f78`.
- Plan semantic SHA: `a8df7f9e0137353baa2218d449558f7977fbe9830f4e89df3856b36e450550ba`.

From the repository root, using CPU only:

```text
python -B -m v3.autodl_supplement_20260908.plan generate --reference v3/autodl_supplement_20260908/plans/reference.json.gz --out NEW_PLAN.json.gz
python -B -m v3.autodl_supplement_20260908.plan verify --plan NEW_PLAN.json.gz --audio-roots AUDIO_ROOTS.local.json
python -B -m pytest v3/autodl_supplement_20260908/test_engine.py v3/autodl_supplement_20260908/test_cloud_pilot.py v3/autodl_supplement_20260908/test_run.py v3/autodl_supplement_20260908/test_score.py -q
```

The original full reference can reproduce the portable projection via `plan export-reference --original ORIGINAL_PLAN.json --out NEW_REFERENCE.json.gz`. Its required original byte SHA is pinned in `plan.py`; ordinary plan generation needs only the versioned compressed reference.

## Execution entry points

`run prepare` verifies all raw audio and base bytes, AutoDL host/GPU identity, actual Linux runtime and source hashes, then writes `SOURCE_LOCK.json`. It requires deployment-local `MODELS`, `AUDIO_ROOTS` and a provision receipt, all generated from verified paths/instance metadata. Real execution rejects Windows or a mismatched host before CUDA use.

```text
python -B -m v3.autodl_supplement_20260908.run prepare --plan v3/autodl_supplement_20260908/plans/PLAN.json.gz --run-dir RUN --models MODELS.cloud.json --audio-roots AUDIO_ROOTS.cloud.json --provision PROVISION.json --source-commit FROZEN_COMMIT
python -B -m v3.autodl_supplement_20260908.run pilot --run-dir RUN --limit 8
python -B -m v3.autodl_supplement_20260908.run admit --run-dir RUN --budget-yuan 80 --gpu-hourly-yuan VERIFIED_PRICE --storage-reserve-yuan RESERVE --already-spent-yuan SPENT
python -B -m v3.autodl_supplement_20260908.run formal --run-dir RUN --limit 8
```

The next batch requires verified off-instance backup acknowledgments once eight units are waiting. A fixed sample keeps its checkpoint permanently. Non-sample weights may be released only with `run release --run-dir RUN --unit-id UID`, after the bound backup acknowledgment exists. A separate local controller owns transfer verification; it never transmits the AutoDL API token into the guest and never scores partial outer results.

`run restore` performs fresh-process GPU inference on AutoDL for fixed samples. It is not optimizer-state training continuation or independent retraining. A failed attempt remains visible; do not erase the attempt or stale lock to restart it silently.

Only after all 720 formal units and their artifact/retention evidence pass:

```text
python -B -m v3.autodl_supplement_20260908.score --plan PLAN.json.gz --run-dir VERIFIED_BACKUP --out NEW_SCORES
```

Source paths in the lock are repository-relative. The archived `RUNTIME_PATHS.local.json` is hashed as original evidence; CPU scoring of a local backup does not need its cloud paths to exist. Keep results, mutable runs, models and audio outside Git. Publish compact verified reports separately.

## Existing diagnostic

[Old-window reconstruction](diagnostics/RECONSTRUCTION_REPORT.md) reproduces the original 15-window selections and preserves all W=8/10/12/15 descriptions. It is explicitly post hoc and is not a result of the new AutoDL training.
