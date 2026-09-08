# SER26-N14R2 complete rerun — results and metadata erratum

## Outcome

The fresh four-RTX-4090 rerun completed all **24 primary draws / 1,920 fits**,
with **zero failed fits, retries or reserve activations**. No old N14R fitted
output was reused. The explicitly labeled metadata-corrected result passed the
unchanged frozen independent verifier with **zero differences**. The original
scorer output did not pass: its one-field reporting defect is preserved and
disclosed below.

The preregistered primary result supports greater validation optimism under
GR than GG for this fixed CREMA-D pipeline. It does **not** establish a loss in
external test performance or isolate a speaker-memorization mechanism.

## Confirmatory result

Both cells use the same paired speaker-disjoint outer test splits. GR uses an
inner stratified sample-random holdout with speaker overlap; GG uses an inner
speaker-group holdout without overlap. Each cell selects among eight frozen
HPO configurations. The primary quantity is

`mean_draw mean_fold{(V−T)_GR_hpo − (V−T)_GG_hpo}`,

where V and T are six-class macro-UAR in percentage points. Five folds are
equally weighted within each draw; the inferential unit is the complete draw.

| Preregistered analysis | Result |
| --- | --- |
| Draw-level mean difference | **+1.419049 percentage points** |
| Two-sided 95% Student-t CI | **[0.901247, 1.936852] pp** |
| Draw-level sample SD; SE of the mean | 1.226256 pp; 0.250309 pp |
| Student t, df | 5.669202, 23 |
| Two-sided p; Holm-adjusted p (family size 1) | 0.000009013204 |
| Number of complete draws | 24 |
| Prespecified directional rule | Supported: mean > 0 and adjusted p ≤ 0.05 |

Prespecified sensitivity analyses were not substituted for the primary test:

- Draw bootstrap percentile 95% CI: **[0.949997, 1.906417] pp**, 100,000
  repetitions, seed 202609050001.
- Complete exact sign-flip enumeration: 80 extreme assignments out of
  16,777,216, two-sided p = **0.000004768372** for both mean and studentized
  statistics. Its symmetry assumption remains necessary.

D09R2 is descriptive only: the HPO-versus-fixed-config change in the GR−GG
outer-test contrast is **−0.075590 pp**, 95% Student-t CI
**[−0.320352, 0.169172] pp**. No confirmatory p-value or verdict is assigned.

## Interpretation limits

This is evidence about validation optimism under the frozen split and HPO
procedures, not a 1.42-point test-accuracy reduction. GR does not leak speakers
into the outer test set. The procedures also change training/validation
membership and potentially sample counts, so the comparison cannot isolate
speaker memorization as a causal mechanism.

The estimand is conditional on the fixed hygienic CREMA-D corpus (7,435 rows),
ResNet-SE, pinned features, eight-configuration grid, algorithmic randomization
and four-node deployment. The 24 draws are not 24 independent corpora; neither
120 folds nor 1,920 fits are treated as independent inferential replicates.
Generalization to other corpora, languages, models or speaker populations
requires further evidence. This rerun does not turn descriptive D09R2 into a
test of external-performance harm or equivalence.

## Reporting defect and transparent metadata-only correction

The unchanged frozen scorer first validates a 28-draw plan (24 primary + 4
reserves), then restricts `by_draw` to the 24 locked analysis draws. Its final
`score.py:1256` incorrectly writes `n_planned_draws = len(by_draw)` as 24.
The independent `verify.py:1292` correctly reconstructs 28. The first full
verification therefore failed at exactly **`$.n_planned_draws`: 24 vs 28**.
This field is assembled only in the output object and does not feed any
selection, statistical calculation or inferential denominator.

The original result and original failed verification are retained byte-for-byte.
A separate `n14r2_results.metadata-corrected.json` changes exactly one byte:
the final digit of that top-level field, **4→8 at zero-based offset 99139**.
No other bytes or JSON values change. No frozen code, PINS, plan, draw set,
training artifact, completion certificate or analysis lock is edited. The
guarded operator script binds the original result, original failure report and
plan hashes, requires the exact singleton mismatch, and exclusively creates
new output files. Its synthetic tests passed (17 passed, one native-symlink
privilege skip; simulated reparse rejection passed).

The unmodified frozen independent verifier then performed its full
reconstruction again in the pinned Linux environment and **passed with zero
differences**. This is a pass of the labeled corrected copy, not a claim that
the original frozen scorer produced a passing file. All original scientific
numbers are unchanged. `n14r2_metadata-erratum.json` records the correction at
creation time; `analysis-verification-receipt.json` records the subsequent pass.
The decision to correct only this structural field preceded interpretation;
the operator receipt also discloses subsequent incidental display of numerical
fields during original-report inspection, after the analysis lock was closed.

## Execution and provenance

- Frozen commit: `4fe0bb52f15b56a6626363be31a378b0f9661293`.
- Preregistration tag: `SER26-N14R2-prereg-1`.
- [Server-timestamped pre-execution receipt](https://github.com/DeerNeverStop/SER/pull/6#issuecomment-5555621048):
  2026-09-05 23:53:20 UTC.
- First formal fit: 2026-09-05 23:54:22.366739 UTC.
- Last node completed: 2026-09-06 03:19:32.060724 UTC.
- Training wall time: **3 h 25 min 9.694 s** (setup, transfer, closure and
  independent scoring/verification are additional).
- Final global stop: 2026-09-06 03:24:34.387768 UTC, followed by blind closure
  and the valid 24-draw / 1,920-unit analysis lock before scoring.
- All four full node backups passed source/local opaque SHA-256 and inventory
  checks. The assembled closed run has 11,592 files and 9,617 directories.
- Three GPUs stopped at approximately 03:26:06 UTC; the retained analysis GPU
  stopped at 03:52:47 UTC after final report download and hash verification.
  GPU stop is distinct from storage deletion; teardown is separately recorded.

Frozen code/environment and actual cloud feature bytes passed complete PINS
checks before and after analysis. The public independent replay records
`external_cache_bytes_verified: false` because the immutable lock contains
Windows cache paths absent on Linux. Actual cloud cache checks are separately
operator-attested; the public bundle does not contain those cache bytes. No
lock paths were rewritten. See `REPRODUCE.md` for the exact replay boundary.

The public archive contains raw prediction/training-output evidence and metadata,
not audio, feature-cache bytes or trained checkpoints. It is an analysis-replay
and audit bundle, not a self-contained raw-audio retraining package. Disclosed
metadata include CREMA-D filenames/speaker IDs, GPU UUIDs, environment details
and two immutable feature-cache paths. Credentials and private operational
configuration are excluded.

## Evidence assets

`n14r2_results.original.json` and `n14r2_verification.original-failed.json`
preserve the original defect. `n14r2_results.metadata-corrected.json` and
`n14r2_verification.metadata-corrected.json` are the corrected-copy/pass pair.
The full unmodified closed run is `n14r2-closed-run.tar` (SHA-256
`162e209301252397c6d7bf79ae8c3a781fca93f8ef7dbe2babace94d4f44f9e9`).
`run-SHA256SUMS` validates its individual files; `RELEASE_SHA256SUMS` validates
release assets. The release tag points to the same frozen preregistration
commit; the report, erratum and outputs are attached assets, not hidden edits
to that commit.
