# DEPLOY-2 enrollment implementation review

2026-09-05. This review covers `enroll.py` and its synthetic tests. It does not
report formal experiment effects, train on real features, or modify DEPLOY-1,
Study II, or N14R results.

## Scientific decisions and fixes

- The shared primary reference budget is N=3 for all corpora and all enrollment
  endpoint families. All feasible N=5 conditions remain secondary; balanced
  N=10/20 absolute and contrast results also remain in the fixed output table.
  The single-emotion average uses the feasible emotions listed in each endpoint,
  without pooling different corpus class sets into one effect.
- `balanced@3` contains the three classes selected by the frozen salted order;
  it is a specific class-dispersed subset. The primary full-class single mean
  versus this subset is descriptive, not a pure diversity causal effect. A
  secondary contrast restricts the single mean to exactly these same three
  classes, matching the average class mixture without extra fits. It is always
  reported and cannot replace the original primary according to its result.
- All speakers must have every fixed evaluation class in Q. Missing support now
  fails planning and execution rather than silently changing the UAR class set.
- Other-speaker references use a deterministic cyclic derangement within the
  outer test fold. A donor's exact same frozen composition/N/repetition/draw
  sample is reused for its recipient. The recipient cannot receive itself, and
  enrollment paths exclude the donor's Q. The previous code seeded own and other
  draws separately, adding an unnecessary sampling difference.
- Main harm share now thresholds each speaker's emotion-averaged single UAR
  minus that speaker's global-standardization UAR. The previous implementation
  instead averaged emotion-specific harm indicators, measuring a different
  quantity. Per-emotion harms remain secondary. Threshold is strictly below
  minus two UAR percentage points, after draw and repetition averaging.
- Reference-emotion recall is reported for every feasible N/emotion, including
  N=5. Its across-emotion average is also emitted; N=3 is primary.
- `oracle_all` is explicitly descriptive and transductive: Q features enter its
  normalization. It is not guaranteed to outperform deployable conditions.
- E2 retains separately linear mean and variance shrinkage with k=8. It is not a
  mixture-moment estimator and intentionally omits the between-mean term. E3
  retains its fixed two-pass rule: global Ridge first, E1-normalized Ridge
  second, top-two neutral pseudo-probability fallback, E2 variance on all refs.
- E4 remains an EM-style heuristic on uncalibrated class-balanced Ridge
  pseudo-posteriors, initialized with utterance-weighted training class priors.
  It does not establish label shift. The numerical policy is explicit: clip
  input pseudo-probabilities to 1e-12, normalize rows, stop when maximum prior
  change is below 1e-8 or after 50 iterations; reject invalid/empty matrices.
  Training emotion offsets remain equal-weight averages over eligible training
  speakers. Reference labels are used to construct conditions, never by the
  estimators themselves.

## Evidence contract

Prediction columns remain unchanged. Each completed unit additionally contains
`references.json.gz`, with schema version, program, module, unit ID, plan digest,
and records `{condition, draw, speaker, source_speaker, paths}`. `model.json`
contains its SHA-256. `DONE` contains prediction SHA-256 then model SHA-256.
Resume verifies the receipt and bound metadata. Completion scoring replays every
reference list from the sealed plan and refuses mismatches. A changed plan
digest or mismatched completed-unit identity is refused. No aggregate UAR,
effect, interval, or harm statistic is printed by the runner.
The plan also freezes an outcome-free endpoint identity manifest (including
unavailable rows); the scorer refuses missing or duplicate endpoints.

## Validation performed

With `E:/科研/SER/ser_gpu/Scripts/python.exe`, CUDA hidden, two CPU threads and
BelowNormal priority: **19 enrollment tests plus 10 managed CPU tests passed**
(29 total, 8.23 seconds). Enrollment tests include deterministic
sampling and predictions; exact shared donor references; Q/E isolation;
reference-label poisoning with unchanged predictions; independently computed
nested averages; primary N=3/N=5 retention; hand-constructed harm-order example;
incomplete runs; receipt tampering even after recomputing all hashes; missing
query classes; invalid EM inputs; and pre-run plan mutation refusal.

A metadata-only audit rebuilt the three corpus plans from the nine original GG
split files and checked all 45 fold/repetition reference manifests. It read no
feature caches or predictions and fitted no model:

| Corpus | Speakers | Feasible conditions | Feasible N=3 single emotions | Query sentences |
|---|---:|---:|---:|---|
| RAVDESS | 24 | 22/40 | 7 | S1 |
| CREMA-D | 91 | 32/32 | 6 | IWW, WSI, ITS, TIE, TSI |
| SUBESCO-980 | 20 | 22/36 | 7 | S9, S1, S8 |

Every query speaker has all evaluation classes. Every other-speaker reference
list matched the donor's own draw exactly, with the declared source identities.
Program-specific salts changed the CREMA-D/SUBESCO query lists from DEPLOY-1;
this is a new disclosed program, not a reinterpretation of its old outputs.
Under the new salted class order, RAVDESS balanced@20 is feasible: neutral is
sixth and receives two references. All three corpora support balanced N=3/5/10/20.

## Managed CPU queue

`python -m v3.deploy.managed_cpu run --deadline 2026-09-09T18:00:00-04:00`
executes the frozen A, B2 Ridge, then B2-FIX queues with the existing local CPU
profile. Output roots are `work/A`, `work/B2_cpu`, and `work/B2_FIXED`.
`work/cpu_control` contains the fsynced attempt journal, outcome-blind status, and
completion receipt. An exclusive workspace lock prevents duplicate managers.
The lock/source/plan/cache/environment are checked before execution; source,
plan, and environment are rechecked before each unit. A interrupted-start event
consumes an attempt. Each unit has at most two attempts across process restarts;
technical failures after both attempts remain incomplete. Integrity failures
stop the queue. Raw DONE recovery requires an existing durable managed start.

Every successful unit gets a hash-bound CPU seal after path, class-label,
identity, reference-source and raw output checks. The completion receipt lists
all unit seals and contains no scientific aggregates. Synthetic tests cover
concurrent launches, transient and persistent failures, restart accounting,
truncated journals/seals, tampering, unmanaged DONE, incomplete closure, source
drift before any attempt, and a real synthetic A model/prediction/receipt
roundtrip with all scientific scorer functions disabled.

## Remaining integration responsibilities

The root task owns final specification, complete-plan/source/input/environment
locking, single-runner control, maximum-attempt policy, cloud allocation and
resource limits. The independent verifier is being updated separately to replay
the new receipt and output endpoints without importing enrollment code. These
integration checks must pass before formal outputs can be accepted.
