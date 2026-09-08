# Common-fit validation-speaker exposure: frozen execution protocol

Study `SER26-DUAL-VALIDATION-1`, designed after the completed speaker-coverage
study. This is a controlled extension of the paper's validation-optimism
question, not an exact replication of N14R2's random/grouped partition program.
Earlier results informed this question; no new scientific results inform this
plan. A private Git commit and a content-hashed plan precede new training; these
are pre-execution freezing, not a claim of public preregistration.

## Question and design

With identical fitting recordings, model initialization, optimization trajectory
and external test, does using validation speakers seen during training yield
greater selected-model validation optimism than using unseen validation speakers?

- CREMA-D inherits the same 7,435-recording hygiene population, 91 people and
  deterministic medium/unspecified-intensity class/text representatives. No new
  participants, recordings, ASV selection, TTS training, or historical weights.
- 24 fresh draw randomizations, five outer folds per draw, stratified by recorded
  sex. Hash ordering in the new program namespace fixes all splits and seeds.
  Test people partition all 91 people within each draw. The corpus is fixed.
- In each draw/fold, choose four fitting texts and two disjoint query texts by
  hash. Among outer-development people with all six text/class cells, select
  24 fitting people (12 female/12 male), then 24 disjoint unseen-validation
  people (12/12), by fixed within-sex hash ordering. All 120 contexts must pass
  capacity checks before training. No score-dependent replacement or reserve.
- Fit: 24 people × four texts × six classes = **576** distinct recordings.
  Seen validation: the fitting people × two query texts × six = **288**.
  Unseen validation: the other 24 people × the same two texts × six = **288**.
  Validation sets have exactly equal sex/text/class counts. Every recording
  role is disjoint. Shared speaker identity is intentional only for fit/seen-val.
- External test includes all outer-fold people on the same two query texts,
  using all available representatives. Missing cells are recorded, not replaced;
  every test person must retain all six classes. Class/text support must be
  audited before training. Query text is shared across both validations and test
  but is absent from fitting; the target includes this cross-text condition.
- Each configuration has **one** fitting trajectory, observed by both validation
  sets. Validation evaluation is deterministic and does not update parameters.
  Thus the two selection rules cannot alter training membership, epoch count or
  optimization steps. Neither validation set is reused for fitting/refitting.

## Model, candidates and count

Reuse the independently byte-verified WavLM-base+ pretrained checkpoint
`136a3e720c04f2c77bf7a4dc6a3868b14d5a2c145a988114b733cb1a8428be98`.
Train its top four transformer layers and a linear six-class head. All parameter
storage is FP32; training uses float16 AMP, inference FP32. Batch 16, AdamW,
weight decay .01, constant learning rates, random three-second training crops
and first-ten-second evaluation. Train all 15 epochs, without early stopping.
Matmul TF32 is disabled; cuDNN TF32 is enabled and recorded with runtime versions.

The candidate grid, in this exact order, is encoder LR {1e-5,5e-5} crossed with
head LR {3e-4,1e-3}. Fixed comparator is index 3, (5e-5,1e-3), inherited from the
earlier WavLM training configuration. No dropout or trainable-layer search.
Paired configurations share the draw/fold training seed; the two validation rules
share each actual fitted trajectory. Class-balanced cross-entropy has all-one
weights on these balanced panels. Validation loss is total CE divided by rows.

**480 formal fits = 24 draws × five folds × four candidates**, not 960 fits.
Four isolated technical pilot fits repeat draw 0/fold 0's grid in a separate
directory and are excluded from scientific analysis. Pilot outputs inspect
feasibility, resources and restoration, not accuracy or favorable candidates.

## Selection, estimates and limits

For each configuration, independently select earliest minimum-loss epoch on
seen and unseen validation. Within each draw/fold and rule, select the candidate
with highest own-validation whole-fold six-class UAR at its selected checkpoint;
ties use lowest candidate index. The same validation thus selects checkpoint
and candidate, and its reuse is part of the measured optimism.

Let V and T denote selected-model validation and common external-test whole-fold
six-class UAR (%). Primary per-draw delta is the equal mean of five folds of
`(V_seen − T_seen) − (V_unseen − T_unseen)`, in percentage points. The estimate is
the mean of 24 complete draw deltas. Freeze a two-sided one-sample t test and
95% t interval on those 24 values (one primary hypothesis); report positive,
negative and inconclusive results alike. A 50,000 draw-bootstrap percentile
interval with seed 2026090701 is sensitivity only. No result-dependent extension.

Draws describe fresh randomizations/training seeds **conditional on this fixed
corpus, pipeline and deployment**, not independent newly sampled populations.
Folds, recordings, candidates and the 480 fits are not independent inferential
replicates. Do not pool these intervals with old speaker-bootstrap or N14R2
intervals. SD=0 produces a disclosed undefined t statistic, not fake certainty.

Report all validation and test scores, selected indices/epochs, and descriptive
decompositions ΔV and ΔT. Include fixed candidate 3 with (a) each rule's best
checkpoint and (b) common last epoch. The latter involves no adaptive selection:
same model + same test implies its ΔT is identically zero and optimism difference
equals V_seen−V_unseen. Primary minus that reference is a descriptive selection
increment, not a causal HPO effect or a new hypothesis test. Fixed-candidate best
still includes checkpoint selection. No equivalence, general language effect,
universal correction, pure voiceprint mechanism, or guaranteed test improvement.

## Execution, retention and release

Use one resumed, already configured Runpod Secure RTX 5090; read current price
and runtime before starting. Prior observed price is USD .99/hour. Initial
working estimate is 5–9 rental hours including repeated validation and backups;
measure four pilot units before launching the full fixed grid. Operator compute
ceiling is USD 15, excluding retained disk fees, not a provider-enforced cap.
If timing makes the grid exceed the ceiling, stop after the pilot and report
feasibility; do not truncate the formal design based on results.

Store predictions for all three checkpoints (seen-best, unseen-best, last) on
both validations and test, plus complete loss history and lossless state deltas.
Deduplicate identical selected epochs; never quantize or omit a required state.
Every stored unique state must actually reload from the exact base and reproduce
all role logits (atol/rtol 1e-5). Record skipped AMP updates, epoch counts, timing,
memory, model/input/source hashes, attempts and failures. No new score is displayed
before the complete formal gate. Integrity verification does not certify scientific
causality or reproduce training on a different hardware/software environment.

Use new execution/output paths and immutable attempts. Stop the affected batch
on a material error/OOM; preserve failed evidence and version any repair before
continuing. Transfers may be batched: only this study's exact files, individually
hash-verified on the local D: archive, can be removed from the cloud to free space.
Keep all original study files, new phase ledgers and cloud run receipts. Seal the
complete grid locally after offload, independently recompute statistical outputs,
publish compact code/results/reports to Git, and explicitly stop the GPU.
