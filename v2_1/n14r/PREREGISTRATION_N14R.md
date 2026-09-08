# N14R preregistration

Status: **frozen before any N14R training or outcome scoring by the exact commit/tag and public pre-execution receipt that reference this file**. Execution is prohibited until `PINS.json`, that remote tag, and the timestamped receipt all exist. The previous v2 result at commit `c0c0beb` is a design pilot only and is excluded from every N14R estimate, test, confidence interval, and decision.

## Question and estimand

N14R asks whether hyperparameter selection using speaker-overlapping inner validation has greater validation optimism than selection using speaker-grouped inner validation, while holding the speaker-exclusive outer test fixed. The target is conditional on the frozen CREMA-D manifest and averages over the prespecified split and training randomization distribution.

For outer draw (d), fold (f), and cell (j\in\{GR,GG\}), select the configuration (c^*_{dfj}) with maximum whole-validation-fold six-class macro-UAR among the eight frozen configurations. Checkpoint selection within a fit is minimum validation loss. Exact configuration ties are resolved by the smallest `config_index`. Let (V_{dfj}) and (T_{dfj}) be validation and outer-test macro-UAR, in percentage points, for that selected configuration. Define

\[
\Delta_d={1\over5}\sum_f\{(V_{df,GR}-T_{df,GR})-(V_{df,GG}-T_{df,GG})\},
\qquad \theta={1\over24}\sum_d\Delta_d.
\]

The independent inference unit is the **complete five-fold outer draw**, not a speaker, utterance, fold, fit, or prediction. Folds are averaged with equal weight within a draw. Validation and test UAR are each calculated once on the entire fold, with the same six frozen classes; per-speaker UAR is not used.

## Frozen design

- Corpus/model: the pinned 7,442-row CREMA-D byte manifest, the preregistered v2 H1/H2/H3 hygiene transform, and the ResNet-SE implementation used by v2. H1 drops both members of label-conflicting exact-byte duplicate groups, H2 retains only the lexically first member of same-label exact duplicates, and H3 drops the registered unusable file. The resulting split/training population is exactly 7,435 utterances, reindexed contiguously; `plan/hygiene.json` records all seven exclusions and reasons.
- Draws: 24 primary IDs `0..23`, followed by four pre-generated reserve IDs `24..27`. Namespace: `SER26-N14R`. No old v2 seed or result is reused.
- Each draw contains five independently randomized, stratified speaker-grouped outer folds. Both cells and every configuration use the identical outer train/test membership in a draw/fold. Speakers are disjoint across outer train and test, and all six classes must occur in each partition.
- `GR_hpo` uses a stratified random 25% inner validation split and must have a nonzero fit/validation speaker intersection; `GG_hpo` uses a speaker-grouped 25% inner validation split and must have zero intersection. Both cells must contain all six classes in fit and validation. The overlap speaker count and fraction are recorded per fold. The grouped rule tries the frozen seed plus offsets `0..99`, choosing the first valid split.
- The grid is the original eight v2 combinations of learning rate `{3e-4,1e-3}`, weight decay `{1e-4,1e-3}`, and dropout `{0.1,0.3}`, in the order frozen by `spec.py`. Batch size 64, maximum 100 epochs, patience 15.
- There is one training replicate. Within each draw/fold the same `train_seed` is shared by both cells and all configurations. Thus the design has 80 unique planned unit identities per draw, 1,920 primary units, and at most 2,240 unique units after reserves, before retries. Each unit has one initial attempt and at most one identical retry, so the theoretical ceiling is 4,480 training attempts.

`unit_id` is SHA-256 of all scientific identity fields: study/spec/manifest, draw membership and analysis order, fold, replicate, cell, model/corpus, split and training seeds, outer/inner/config hashes, sample counts, namespace, and full configuration. Only run status and estimated cost are excluded.

## Confirmatory inference

The sole member of confirmatory family `N14R` is tested with a two-sided one-sample Student t test of the 24 draw-level deltas against zero at alpha .05. A directional success requires both `mean(delta)>0` and the two-sided Holm-adjusted p-value `<=0.05`. Family size is one, so Holm-adjusted and raw p-values are identical. Report the mean, sample SD, SE, t statistic, df=23, two-sided p-value, and two-sided 95% Student-t CI.

Two prespecified sensitivities do not replace the primary test: (i) a percentile bootstrap over the 24 draws with 100,000 resamples and seed `202609040001`; and (ii) the full `2^24` sign-flip distribution. The latter is described as exact only conditional on exchangeability under sign reversal (symmetry), not as a distribution-free test of a mean.

The design alternative is 1.5 pp with draw SD 2.5 pp. A two-sided noncentral-t calculation at alpha .05 and n=24 gives power 0.804. These values were rounded conservatively from the excluded pilot and are not revised after N14R begins.

## Completion, missingness, and stopping

A draw is complete only when all 80 planned units are valid: five folds, both cells, and all eight configurations. Each unit may be retried once with the exact same configuration, split, and seeds. A second failure voids the entire draw; a result may never be selected from a reduced grid. Reserve draws replace void draws in ascending ID order only. The analysis set is the first 24 complete draws in frozen `analysis_order` among primary and activated reserves.

No effects, confidence intervals, p-values, or selection summaries may be computed before the completion ledger is locked. There is no efficacy or futility stop. At most 28 draws may be attempted. If fewer than 24 draws are complete after the prescribed retries and reserves, N14R is reported as `not tested (incomplete)` and only clearly labeled descriptors may be shown; no p-value placeholder is manufactured.

The no-failure projections are about 92.9 GPU-hours for the primary draws and 108.4 GPU-hours if all 28 draws receive one attempt. They exclude retries. Retrying every one of the 2,240 possible units would project to about 216.8 GPU-hours. The 125 GPU-hour figure is a soft operational allocation, not a hard cap, completion guarantee, result-dependent rule, or scientific stopping rule. The fixed attempt/reserve rules control stopping, and actual attempts and GPU-hours will be reported.

## Secondary descriptor D09R

Configuration index 4 is the unchanged fixed comparator. For each draw, compute the equal-fold mean

\[
\Gamma_d={1\over5}\sum_f\{[T_{GR}(c^*_{GR})-T_{GG}(c^*_{GG})]-[T_{GR}(4)-T_{GG}(4)]\}.
\]

Report the mean Gamma and a 95% Student-t CI over draws. D09R has no p-value, multiplicity-adjusted decision, or confirmatory verdict. N14R identifies validation optimism; it does not by itself establish outer-test performance harm.

## Artifact and audit rule

The machine authority is `spec.json` plus generated `hygiene.json`, `run_plan.csv`, `split_index.json`, `unit_configs.json`, and 56 cell-specific split JSON files. Outputs use the existing trainer layout `runs/n14r/units/U/{unit.json,history.json,predictions.csv,DONE}`. Before fitting, the runner verifies the complete `PINS.json` inventory, exact package lock, GPU/runtime identity, unique feature-cache candidate, cache schema/content, hygienic manifest coverage, and every split population. Test UAR is recomputed from raw outer-test predictions; selected validation UAR and epoch must match the minimum-validation-loss row in the full history. Split/prediction coverage, labels, identities, all hashes, six-class support, shared outer tests, complete grids, equal fold weighting, tie-breaking, pilot exclusion, and ledger lock are fail-closed verifier checks.
