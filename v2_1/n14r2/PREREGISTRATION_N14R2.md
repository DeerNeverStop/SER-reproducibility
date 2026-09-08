# N14R2 prospective full-rerun preregistration

This document becomes a frozen preregistration only through the exact pinned
commit/tag and a server-timestamped public pre-execution receipt. Creating this
file alone does not establish a freeze. No formal N14R2 fit may be dispatched
until that receipt, the pins, the complete generated plan, and all four validated
environment receipts exist. The contract version is `2.0.0`.

## Scope and exclusions

N14R2 is a fresh study (`N14R2`, seed namespace `SER26-N14R2`), not a resumed or
repaired ledger for N14R. N14R ended `not_tested_incomplete` after an infrastructure
incident and the original frozen failure policy. All 16 completed N14R draws,
all 1,298 successful N14R fits (including 18 from a partial draw), and every other
N14R output are excluded from N14R2 estimates, uncertainty intervals, selection,
and decisions. No old fitted weights, unit results, split draws, or training
seeds are imported. The earlier v2 pilot at `c0c0beb` remains design-only and is
also excluded. No N14R effects were inspected to choose this rerun design.

The same raw CREMA-D manifest, deterministic hygiene transform, feature cache,
and public model/trainer source may be reused as input materials. These are not
fitted outputs. The original frozen N14R files and failed-attempt ledger remain
unchanged and auditable; this study does not erase or retrospectively relabel
that incomplete study.

## Question and estimand

Does hyperparameter selection using speaker-overlapping inner validation have
greater validation optimism than speaker-grouped inner validation when the
speaker-exclusive outer test is held fixed? The target is conditional on the
fixed corpus, frozen pipeline, and prespecified randomization distribution; it
is not inference over new corpora or an unrestricted speaker population.

In draw d, fold f, and cell j in {GR, GG}, select the configuration c* with the
largest whole-validation-fold six-class macro-UAR from the eight frozen
configurations. Break exact ties by the smallest `config_index`. Within each
fit, the retained epoch is the minimum-validation-loss checkpoint, as in the
frozen trainer. Let V and T be the selected model's whole-fold validation and
outer-test macro-UAR in percentage points. Define

\[
\Delta_d=\frac{1}{5}\sum_f\{(V_{df,GR}-T_{df,GR})-(V_{df,GG}-T_{df,GG})\},
\qquad \widehat\theta=\frac{1}{24}\sum_d\Delta_d.
\]

The inference unit is a complete five-fold outer draw. Folds receive equal
weight within a draw; folds, speakers, utterances, fits, and predictions are not
treated as independent inferential replicates. Whole-fold macro-UAR uses the
same six classes and is not an average of per-speaker UARs. Independent draw
seeds generate complete five-fold partitions; the five folds within one draw
are not themselves independent draws.

## Scientific design retained from N14R

- The pinned raw CREMA-D manifest has 7,442 rows. The original H1/H2/H3 hygiene
  rules exclude seven files and leave 7,435 contiguous, reindexed utterances.
  The generated `plan/hygiene.json` records the exclusions and reasons.
- Primary draw IDs are `0..23`; reserve draw IDs are `24..27`. Every draw and
  every seed is newly generated under `SER26-N14R2`. All 28 draws are planned
  and pinned before formal execution, irrespective of reserve activation.
- Every draw uses a randomized stratified speaker-grouped five-fold outer
  partition. Both cells and all configurations share the identical outer
  train/test membership in each fold. Outer train/test speakers are disjoint,
  and each partition contains all six classes.
- `GR_hpo` has stratified random 25% inner validation with nonzero fit/validation
  speaker overlap. `GG_hpo` has speaker-grouped 25% inner validation with zero
  overlap. Each inner partition contains all six classes. The grouped rule
  tries its seed plus offsets `0..99`, taking the first admissible partition.
- The grid order is learning rate `{3e-4, 1e-3}`, then weight decay
  `{1e-4, 1e-3}`, then dropout `{0.1, 0.3}`. The ResNet-SE `p1_frozen` model,
  augmentation, optimizer, scheduler, checkpoint selection, and early stopping
  remain as pinned in the original source. Batch size is 64, maximum epochs
  100, and patience 15. Fixed comparator index 4 is `(1e-3, 1e-4, 0.1)`.
- One training replicate is used. Both cells and all eight configurations
  share a training seed within each draw/fold. There are 80 unique units per
  complete draw, 1,920 primary units, and at most 2,240 units with all reserves.

Seeds are the first four SHA-256 bytes (unsigned, big endian) of the explicit
outer/inner/train domain strings in `seeds.py`. The bootstrap seed is
`202609050001`. Unit identity is SHA-256 of every plan field except `unit_id`,
`est_gpu_sec`, and `status`, plus the namespace and full configuration. Thus
study/version, plan order, all seeds, partitions, input/config hashes, and sample
counts are bound; operational estimates and progress are not scientific identity.

## Four-node execution fixed prospectively

There are four separate Linux pods, each with one NVIDIA GeForce RTX 4090 and
eight spawned persistent fit-worker processes. Each worker uses one PyTorch CPU
thread, resets all training RNGs before every unit, and fits an independent
model. This is task parallelism, not distributed-gradient training. Training
uses FP32 tensors without autocast or GradScaler; the original backend TF32
settings are retained, so this is not a promise of all-IEEE-FP32 arithmetic.
There is no epoch-checkpoint resume: a retry starts the same unit
from initialization with its original configuration, split, and seed.

The full draw is assigned by `node_index = draw_id % 4`; every paired cell and
configuration of that draw remains on that node/GPU/runtime. The 24 primary
draws give six full draws per node. Independent fits within a draw can execute
concurrently. Node assignment is fixed before outcomes; a fast-finishing node
does not steal another node's draw or activate its own reserves.

The exact core environment is Linux, Python 3.12.3, PyTorch 2.11.0+cu128,
CUDA runtime 12.8, cuDNN 91900, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0,
librosa 0.11.0, and RTX 4090. Each node's full package, GPU, driver, CPU,
environment identity, source, manifest, and cache receipt is bound before
execution. Hosts/drivers can differ between nodes even when these core versions
match. The environment receipt records CPU architecture and thread settings;
the receipt together with the pin inventory binds the sources and inputs.
Pairing within one node reduces this issue, but does not prove bitwise
equivalence or eliminate host variation. The reported estimand and limitations
must acknowledge this fixed four-node deployment; node-specific results cannot
be used to revise inclusion or allocation.

## Attempts, circuit breaker, and reserve state machine

An actual attempt is conservatively charged when a healthy process has an
immutable attempt directory reserved and its unit is dispatched. It counts even
if a later interruption leaves it uncertain whether training actually began.
Worker initialization and health checks without dispatch are not attempts.
Every unit permits at most two charged attempts. Failed, interrupted, and
successful attempt provenance is append-only; no deletion or reset restores
attempt eligibility.

An infrastructure, CUDA, worker-death, timeout, or unclassified failure pauses
dispatch and trips the node circuit breaker. Other already-dispatched work is
accounted for; it is not silently erased. No infrastructure failure automatically
voids a draw or activates a reserve. An eligible retry requires global recovery
authorization, a fresh process, and a successful health check before dispatch.
The same seeds and configuration must be retained. One failed process must not
continue consuming remaining units or reserves with a poisoned CUDA context.

Only two explicitly allowlisted, outcome-blind training-level failures may void
a whole draw under the frozen implementation. The allowlist and classification
code are pinned before the first fit; observed performance or effect direction
is never a failure class. The allowlist is `FloatingPointError` after CUDA and
infrastructure exclusions, or the exact lowercased exception message
`no validation checkpoint`. Two infrastructure failures, mixed failure classes,
or exhausted unknown failures leave the unit/draw **blocked**, not void. Such
an exhausted infrastructure block cannot be converted to a reserve under this
contract. A change requires a prospective public amendment and user direction
without reading outcomes; it cannot be presented as unmodified N14R2 compliance.

All 24 primary draws must first reach `complete` or training-failure `void`
(`void_training_failure` is the semantic alias, not an additional failure rule).
Paused/blocked/pending draws do not satisfy this barrier. If fewer than 24 are
complete, a single global writer may authorize the next reserve in ascending
order `24,25,26,27`, one at a time and only as needed. Every reserve uses the same
80-unit completeness, attempt, and failure rules and the fixed `draw_id % 4`
assignment. Nodes cannot activate reserves from local completion alone. A draw
is complete only when all 80 immutable unit receipts are present and valid;
there is no reduced-grid or partial-draw analysis. The analysis set is the first
24 complete draws in frozen `analysis_order` among the primaries and authorized
reserves, never the first 24 to finish by wall-clock time.

The theoretical maximum is 4,480 charged attempts (2,240 units times two), not
an authorization to continue through an infrastructure incident. The USD 25
agent guard pauses further dispatch and requests direction; it is neither an
efficacy/futility rule nor a scientific completion or stopping boundary. Cost
or runtime predictions are operational estimates, not promises. The CSV's
`est_gpu_sec=0` means unestimated, not zero cost. There is no hard scientific
runtime cap. A pause or block is not declared terminal incomplete merely to
produce a report.

If fewer than 24 draws can be completed after all eligible retries and four
authorized reserves under the unchanged rules, the terminal scientific status
is `not_tested_incomplete`. No primary p-value, confidence interval, selected
configuration summary, or effect estimate is manufactured. Before an analysis
lock exists, close-out reports may contain only outcome-blind completeness and
incident descriptors. Any subsequent substantive amendment is clearly labeled
and cannot silently overwrite this registration.

## Confirmatory inference and descriptor

The one-member corrective family `N14R2` tests a zero mean draw delta using the
two-sided one-sample Student t test at alpha .05. A directional success requires
both a positive mean and Holm-adjusted two-sided p <= .05. Family size is one,
so adjusted and raw p-values coincide. Report mean, sample SD, SE, t, df=23,
two-sided p, and a two-sided 95% Student-t CI. The approximation is over the
prespecified draw-level pipeline randomness, not 120 independent folds or
1,920 independent fits. Cross-node numerical variation is a limitation, not
grounds for outcome-dependent exclusion.

Sensitivity analyses are a 100,000-resample draw percentile bootstrap with
seed `202609050001`, and full enumeration of the `2^24` sign assignments.
The latter is exact only under sign-exchangeability/symmetry, not for an
unrestricted mean-zero null. Neither sensitivity replaces the primary test.
The inherited design alternative 1.5 pp, draw SD 2.5 pp, n=24, and two-sided
alpha .05 imply noncentral-t power approximately .804. This is an excluded-pilot
design assumption, not a guarantee or an update from N14R outcomes.

The descriptor `D09R2` is the mean over complete draws of the equal-fold mean
`[T_GR(selected)-T_GG(selected)]-[T_GR(index4)-T_GG(index4)]`. Report its mean and
95% Student-t CI only, with no p-value or confirmatory verdict. The primary
study addresses validation optimism; it does not by itself establish outer-test
performance harm. The historical studies are disclosed, but their observations
are not pooled with this new prospective family or portrayed as independent
successful replications.

## Artifact and outcome-blinding contract

`spec.json` must exactly equal the serialized Python `SPEC`. The pin inventory
also binds the generated hygiene/plan/config/split files and all scientific and
execution sources. Split keys are `n14r2__dDD__CELL`. Attempts live under
`runs/n14r2/nodes/nodeN/units/U/attempt_01` or `attempt_02`; the unit's
`DONE.json` points to one immutable successful attempt. Per-attempt
`artifact_receipt.json` contains identity, hash, and coverage/completeness
metadata but no performance metric. Single-writer node ledgers and the global
authorization ledger preserve the full attempt and reserve provenance.
The global ledger is `runs/n14r2/global_ledger.jsonl`; the analysis lock is
`runs/n14r2/analysis_lock.json`.

Before the analysis lock, the coordinator and close-out checker may inspect
only outcome-blind metadata and hash artifact bytes; they must not parse unit
metrics, histories, predictions, select configurations, or aggregate effects.
An analysis lock requires all 24 accepted draws and binds the four original
environment receipts, node/global ledgers, every attempt, the selected immutable
success paths, and the exact plan/pins. After that lock, scoring and independent
verification check identities, coverage, six-class support, checkpoint/history
agreement, raw outer-test predictions, whole-fold UAR, the complete eight-config
grid, paired outer tests, tie-breaking, and equal-fold/draw aggregation. No
effect-based interim scoring, efficacy/futility stopping, or post-outcome
alteration of this contract is permitted.
