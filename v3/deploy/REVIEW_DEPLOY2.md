# Independent pre-execution review: SER26-DEPLOY-2

2026-09-05. Scope: scientific assignment, training leakage, artifact integrity, and independent numerical verification. No real predictions were scored or inspected. No experiment was launched by this reviewer.

## Decision

The revised design is reasonable for a descriptive controlled study. **No remaining scientific veto to the locked experiment** was identified. Formal launch still requires the execution owner's complete test run, final code/plan lock, independent CPU/GPU environment receipts, verified cloud inputs, and the predefined blind timing/budget gate. Passing this review does not mean formal results already exist or have been independently verified.

The metadata-only receipt is `work/metadata_independent_audit.json`. It includes the three plan file hashes and both verifier source hashes. It confirms:

- A: 135 units, 91 CREMA-D / 24 RAVDESS / 20 SUBESCO speakers; respectively 32 / 22 / 22 feasible reference conditions; every query speaker retains the fixed class set; query/reference sentences are disjoint; the new program's sentence hash is independently reproduced.
- B2: 390 planned units and 195 model-expanded GR/GG pairs. Every pair has identical outer-test recordings and different fit recordings. All 18 source split files match the source index hashes. Test speakers are separate from fit/validation, GG validation speakers are unseen, and GR validation speakers have the intended fit overlap.
- B2-FIX: 135 units over 45 independently reconstructed matched partitions. Donor selection, half-recording eligibility, donor-pair/sentence/class matching, whole-cell drops, retained fit recordings, four-way recording exclusion, calibration speaker counts, and source GG test assignments all pass.

## Scientific changes required and completed

1. B2 is the effect of the entire training/validation protocol, including early stopping. It cannot isolate calibration overlap because all paired fit sets differ. B2-FIX shares one fitted model and one test prediction file and substitutes only calibration sources.
2. A's shared N=3 control separates reference count from composition. Its broad single-emotion average and its fixed three-class dispersed comparator have different average class compositions; the additional frozen balanced-class-matched secondary contrast makes that distinction testable. Neither contrast is a general optimal-enrollment recommendation.
3. Reference labels construct the controlled stress conditions, while estimator fitting/prediction does not consume their labels. E2 is linear variance shrinkage, E4 uses pseudo-posteriors with unverified label-shift assumptions, and the query-feature reference is not a guaranteed upper bound.
4. Calibration coverage gaps use actual calibration coverage. Batch-budget curves reject exactly floor(b*n), with deterministic path-hash ties, whereas deployed thresholds preserve test-budget drift. Zero acceptance has undefined risk, never zero risk. The secondary threshold scans realizable tied confidence blocks and rejects all with risk NA when infeasible.
5. Paired calibration comparisons preserve pairing within each repeat. Undefined risks propagate through the strict repeat mean. Bootstrap indices use the full speaker frame shared across strategies; undefined resamples and speakers are disclosed. Per-repeat coverage counts and pooled recording risks are distinct from profiles of averaged coverage and speaker-equal risks.

## Runtime findings required and completed

- WavLM previously initialized the random classification head before setting its seed. The new engine seeds before encoder/head creation. A synthetic test exercises the real FT loop with a tiny fake pretrained encoder and proves identical outputs despite different prior RNG states.
- Both engines restore the best validation checkpoint and now return its state for persistent checkpoint/hash receipts. CNN checkpoint reload reproduces exported test logits exactly. Test-label access guards cover both real training engine paths; neither engine requests test labels.
- Formal execution must retain audio and initial WavLM weight hashes, lock the source/config/input chain, and use managed entries for attempts, environment identity, exclusive execution, and sealed outputs. The direct technical runner remains suitable only for the disclosed pilot.

## Independent verifier and tests

The inherited `verify.py` fixture tests used a synthetic scorer whose output schema differed from the actual A/B1/B2 scorers. Those passing tests alone could not substantiate end-to-end scientific verification. `verify_deploy2.py` now supplies the actual DEPLOY-2 path, with no imports from enrollment, calibration, model runners, or scientific scorers. It reconstructs statistics from raw predictions and independent confusion counts. Display/identity mappings are separate from numeric rules.

The final targeted command passed **38 tests in 14.24 seconds**: 21 new DEPLOY-2 tests plus 17 inherited fixture/mutation tests. It ran on CPU with at most two threads and no downloads. Coverage includes:

- Full synthetic A training → path receipts/raw predictions → all independently enumerated endpoint estimates, intervals, and speaker rows, including matched-class secondary; omitted endpoints and changed donors fail.
- Actual B1 scorer versus independent complete summary/class-mix/quota/fold-sensitivity and bootstrap calculations; deliberately changed rejection counts fail.
- Actual B2 scorer versus independent complete nested absolute quantities, raw counts, pooled-by-repeat values, speaker rows, all paired endpoints and interval fields; perturbed intervals and test-derived thresholds fail.
- Actual FIX scorer versus independent unit quantities and all four paired endpoint families; incomplete repeats, swapped calibration sources, leaked recordings, lost donors and unmatched cells fail.
- Tied quotas, secondary cutoff off-by-one, infeasible thresholds, test-label independence, full-frame bootstrap with undefined speakers, FT random-head determinism, CNN checkpoint replay, and duplicate artifact-location rejection.

The verifier accepts either one consolidated B2 tree or managed `B2_cpu`/`B2_gpu` trees, and refuses two copies of the same unit. FIX's plan filename is `B2_FIXED.json`, module identity is `B2-FIX`, and managed raw directory is `B2_FIXED`. Scientific verification must occur after all planned new training closes, using the final artifacts returned from the cloud. Real-result numeric verification remains pending by design.

## Interpretation and operational limits

The shared finite acted-speech populations, prior outcome exposure, repeated training partitions and fixed thresholds limit inference. The intervals do not cover new model fitting, recalibration, or real-call distribution changes. Reporting every frozen condition, failed unit, infeasible cell and non-improving contrast remains mandatory.

Cloud transfer/readiness, GPU compatibility and the total 90-CNN plus 30-FT wall time are operational gates for the execution owner. A training deadline alone does not stop Runpod billing; the control process must stop the pod after backup or at its budget/deadline rule. No cloud cost or completion-time claim is established by CPU synthetic tests.
