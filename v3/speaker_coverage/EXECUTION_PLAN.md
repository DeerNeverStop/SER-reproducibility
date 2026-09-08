# Speaker coverage: execution revision 1, 2026-09-06

This document supersedes operational choices in `docs/research-plan-20260906/`,
which is retained as the original, hash-checked review packet. It is an experiment
to test a hypothesis, not a promise that coverage reduces the generalization gap.
The immutable generated plan, source hashes, independent gate and execution
receipts specify the actual run. Results cannot retrospectively change this plan.

## Questions and scope

1. **E0: Is the speaker-distance measurement stable enough to interpret?** Use
   frozen official ECAPA and x-vector checkpoints on byte-verified CREMA-D audio.
   Report reference-half stability, nearest-three overlap and agreement between
   the two encoders. These are measurement diagnostics, not SER outcomes and not
   a universal 90% accuracy gate.
2. **E1: Does distance to the actual training speakers relate to unseen-speaker
   SER accuracy?** Reuse the existing Study II CNN B=576, S=48 prompt-new runs,
   without retraining. Construct references from each run's permitted training
   prompts, aggregate over the actual six query rotations and three draws, and
   report the continuous association. This is observational, not evidence that
   speaker identity causes the difference. Reusing previously scored per-speaker
   data must be disclosed; it is not a fresh raw-logit replay.
3. **E2, primary intervention: Does choosing training speakers by coverage change
   unseen-speaker SER accuracy at a fixed real-audio budget?** Compare uniform U,
   representative R and coverage C with identical class/sex/text budgets. The
   primary contrast is CNN C minus U in mean speaker UAR, estimated two-sided.
   A small, zero, negative or positive effect is a valid result. R minus U is
   secondary. WavLM fine-tuning is a required second-model check, and frozen
   WavLM Ridge is a sensitivity baseline.
4. **E3: Can controllable synthetic speech provide a valid later intervention?**
   Only a 100-attempt technical probe is authorized for this execution revision.
   Formal E3 requires independent human emotion and identity validation before
   training. Four query prompts (Q4), disjoint from eight allowed training
   prompts, replace the old two-query-prompt proposal; the primary future
   coverage contrast is F minus E. Synthetic reference
   recordings must not include stop/test speakers or query prompts. Reuse a
   synthetic item only when every reference, text, generator revision and
   generation parameter is identical. Probe success alone does not pass E3.
5. Cross-language causality, a new large dataset, G (controlled speaker exposure)
   and E4 are subsequent studies. Language is confounded with corpus/channel in
   the current three corpora. Neither these experiments nor one successful
   synthetic probe establishes a dataset that eliminates all speaker gaps.

## E2 fixed design

- CREMA-D metadata and exclusions are inherited unchanged. All 7,435 retained
  audio files must match the original manifest; the seven excluded files need not
  exist on the execution machine. Old results and source code remain untouched.
- Five speaker folds and the old `roles(meta, 0, fold)` stop split are fixed before
  any arm is drawn. Stop has eight people, four per recorded sex. Draw does not
  change stop or test membership.
- Each of six rotations has eight allowed training prompts and two disjoint
  query prompts under the existing prompt schedule. Stop uses six of the training
  prompts. Sharing training text with disjoint stop speakers is intentional;
  query text and test speakers cannot enter training or selection.
- Eligible training candidates need all required class/text cells. For each
  rotation, each candidate's centroid uses exactly the eight allowed neutral
  representatives: normalize each embedding, average, normalize the centroid.
  Stop/test embeddings and query prompts never affect selection. Diagnostics may
  inspect held-out embeddings afterwards but cannot adapt the selector or grid.
- Select 24 people (12 female, 12 male), four prompts per person, all six classes:
  576 real recordings. Each training prompt has 12 people (six per sex). A paired,
  hash-determined abstract prompt layout is reused across policies. Three draws
  vary the permitted sampling/layout seeds. Layout retries can enforce quota
  validity and distinct fits only, with a fixed maximum of 100 attempts.
- **U:** uniform random selection without replacement within each sex.
- **R:** fixed hash first speaker, greedy reduction of mean nearest-one distance
  over the eligible pool, then same-sex one-for-one swaps until no strict
  improvement or 100 swaps. This is a heuristic, not a global-optimum claim.
- **C:** fixed hash first speaker, quota-constrained farthest-first traversal.
  Its geometric target is worst nearest-one pool distance, not mean SER accuracy.
  It has no claimed unconstrained k-center approximation guarantee under quotas.
- R/C membership and tie order depend on fold/rotation/policy, not draw. They may
  overlap substantially. Report overlap and both mean/max distance, including
  nonselected-candidate diagnostics. Overlap is not a reason to suppress a result.

## Training and counting

| Block | Grid | Formal fits |
|---|---|---:|
| CNN | 5 folds × 6 rotations × 3 draws × 3 policies | 270 |
| Frozen WavLM Ridge | same | 270 |
| WavLM fine-tuning | 5 folds × 6 rotations × draw 0 × 2 training seeds × 3 policies | 180 |
| Total | no optional omission of FT after seeing CNN | 720 |

Technical pilots are isolated extra runs: CNN/Ridge each run fold 0, rotation 0,
all three draws and policies (18 fits total), plus one FT run at fold 0, rotation
0, draw 0, seed index 0, U (19 total). Pilot predictions are sealed and excluded from formal
inference. Pilots inspect feasibility, hashes, epoch counts, memory and runtime;
they do not select a scientifically favorable arm or checkpoint rule.

CNN runs all 100 epochs, batch 32, learning rate 0.001, weight decay 0.0001,
dropout 0.1. FT runs all 15 epochs, top four WavLM layers trainable, batch 16,
encoder/head learning rates 0.00005/0.001, weight decay 0.01, training crop three
seconds, evaluation cap ten seconds, float16 AMP. These inherit the old model
architectures, not old early stopping. The best minimum stop-loss checkpoint
and the last epoch both produce sealed test predictions; no test score selects
a checkpoint. An OOM or material correctness error stops the block for a versioned
repair; it does not silently alter batch size or hyperparameters.

Each successful unit retains input/source identity, complete epoch history,
prediction arrays, checkpoint and hashes. FT stores every trainable parameter
and model buffer losslessly as a delta from the exact byte-hashed base checkpoint;
frozen parameters are checked unchanged. Both best and last deltas are retained;
restoring each from the base must reproduce its respective logits within the
declared numerical tolerance. This is a storage
optimization, not lower-precision checkpointing.

## Analysis frozen before E2 outcomes

No new E2 scientific scores are displayed until the complete designated formal
blocks have passed artifact verification. Report all 720 fits; failures and
missingness cannot be omitted or replaced because of score direction.

Combine each speaker's predictions across the six disjoint prompt-query rotations
within each draw before calculating six-class macro recall (UAR), then average
across draws within speaker (FT instead averages its two training seeds on the
fixed draw-0 panel). Give each speaker equal weight. Report paired C−U
and R−U differences in percentage points, policy means and lower-quartile speaker
UAR. Best checkpoint is primary; last epoch is a prespecified sensitivity result.
Report draw variability and per-fold summaries descriptively. Five folds times
three draws are not 15 independent samples; individual clips are not independent
subjects either. Any paired speaker bootstrap interval conditions on the fitted
models, their shared training data and this corpus. It is not a confidence claim
over all languages, corpora or possible retraining outcomes. Do not infer a causal
mediation mechanism from distance/UAR correlation or geometric target changes.

E1 correlations, x-vector agreement, distance tails and best/last sensitivity
are secondary diagnostics. Do not promote whichever comparison is significant
to the primary endpoint. Do not prewrite a one-point ceiling, run post-hoc power
as evidence of a null, or treat a wide interval as proof of equivalence.

## Cloud execution and gates

The first round uses one resumed Runpod Secure RTX 5090, 32 GB VRAM, observed rate
USD 0.99/hour. Existing audio, frozen feature caches and WavLM weights are reused
only after hash checks. AutoDL is connected, but the exact price/environment for
its programmatic Pro creation path was not verified; a public website price is
not a confirmed quote for that API. No second GPU is started automatically.

Working compute cap for this round is approximately USD 30 (about CNY 200; not
an exchange-rate quote). Measure pilot time before estimating the formal batch.
This cap excludes existing disk-retention charges and is an operator limit, not
a provider-enforced budget setting. Keep execution in bounded processes, monitor
cost/runtime, copy and hash-verify the new evidence, then stop the pod explicitly.
The 180 FT units may be executed in three fixed 60-unit shards, preserving plan
order and all planned units. Once each shard is independently hash-verified on
both endpoints, its committed cloud file copies may be removed to free space.
Only exact backed-up paths may be removed. All phase receipts and the cumulative
attempt ledger must also be backed up. Final block sealing is performed on the
complete local 720-unit collection, not on a temporarily incomplete cloud shard.
Stopping a Python process or an SSH session does not stop GPU billing.

Required gates: independent raw/cache/model/source checks; selector/role/quota
verification; all pilot epochs and prediction/checkpoint identities; full formal
artifact counts and restoration checks; local verified backup; then scoring and
reporting. Unit tests exercise leakage, quotas and failure handling. They cannot
prove the entire study error-free. Deviations are recorded before affected runs
restart, and all superseded attempts remain distinguishable.

## E3 probe acceptance boundary

Use an official, revision-pinned generator with supported English voice/emotion
controls in an isolated environment. Log all 100 attempts including failures,
duration, GPU memory, waveform checks, ASV similarity and generation parameters.
An ASV match alone is not proof of preserved identity, and an existing SER model
cannot supply independent emotion ground truth. At least three independent human
raters and a prespecified agreement/quality rule are required for the later
emotion/identity gate. Report the technical probe separately and stop short of
formal synthetic-arm training if those human judgments are absent.
