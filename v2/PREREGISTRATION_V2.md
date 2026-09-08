# Preregistration v2 (candidate for tag SER26-prereg-1)

Status: **rc2 — frozen at tag `SER26-prereg-1` (2026-09-03, author's RTX 5070 machine)**; rc1 was the GPU-less review candidate with a synthetic SUBESCO placeholder. This file plus the machine-readable registry, split
tables, run plan, scorer, verifier and mutation battery constitute the preregistration.
Nothing in `registry/`, `plan_rc2/` or `ser_v2/` may change after tag-1 except through an
entry in `runs/<run_id>/deviations.jsonl` cited by number.

## 1. Thesis and scope
Speaker-nonexclusive evaluation is common in public SER code and inflates reported UAR.
Four headline claims (C1 audit prevalence; C2 premium and its outer/inner decomposition;
C3 mechanism; C4 modern-pipeline robustness) and four supplement claims (S1-S4), each bound
to registered hypotheses in `registry/hypothesis_registry.csv` and rules in
`registry/claim_map.json`. Forbidden claims are listed in the claim map.

## 2. Populations and hygiene
RAVDESS speech (24 speakers), CREMA-D (91), SUBESCO (20; the 980-utterance repetition-clean
panel is pinned by `registry/subesco_980_pinned.json`; the full corpus manifest is added by
the author before tag-1). Hygiene rules H1-H3 (`ser_v2/corpora.apply_hygiene`) are applied
before any split; the log is `plan_rc2/hygiene_log.json`.

## 3. Seeds and splits
All randomness derives from SHA-256 keys (`ser_v2/seeds.py`); training seeds carry no model,
cell or protocol token. Cells RR/RG/GR/GG (outer x inner), sentence-, take-grouped, LOSO and
size-matched LOSO, the speaker x prompt checkerboard with two prompt-group rotations and
matched training size, panels, and the P1 frozen splits are produced by `ser_v2/splits.py`
with pre-fit assertions. Split tables are pinned by `plan_rc2/split_index.json`.

## 4. Units and budget
`plan_rc2/run_plan.csv` enumerates every unit (unit_id = SHA-256 of its configuration and
split hash). Planned GPU: 38.15 h + 8.0 h conditional (CREMA-D fine-tune seed 1); program cap
50 h; per-arm caps and the truncation order are in `registry/arms.json` and `plan_summary.json`.
Truncated units enter as `not tested (truncated, deviation entry N)`, never `not supported`.

## 5. Hypotheses, families, power
28 hypotheses in claim-wise Holm families (N-decomp 4, N-mech 4+2 estimates, N-modern 3
estimates, N-supp 2, R 11, T 2), alpha 0.05 within each family; no program-wise error rate is
claimed. A priori status comes from `registry/power_table.json` (simulated Holm power under
stated per-speaker effect/SD assumptions; power < 0.5 -> estimate). Unit of inference is the
speaker; per-speaker OOF UAR; averaging over replicates and model levels precedes differencing;
10,000-rep whole-speaker percentile bootstrap (seed 20260903); two-sided Wilcoxon; sign test
when > 20 % ties; TOST for T. Exact procedures: `SPEC_SCORING_CONTRACT.md`.

## 6. Gates (all before tag-1)
G1 registry validates (every hypothesis bound to a claim). G2 split-table assertions pass on
the real manifests. G3 synthetic dry run: scorer and independent verifier agree to 1e-12 on
every reported value; mutation battery rejects every case. G4 runner executes the frozen P1
engine, the probe, Ridge and (on the training machine) the WavLM engines on synthetic inputs.
G5 the audit frame/lineage rule, endpoint dictionary and rater form are frozen and the beacon
pulse is the first after the tag-1 timestamp. Tag-2 (before any FT/HPO/MECHID unit) records
only the off-family timing probe and the derived FT cap; it cannot alter anything else.

## 7. Execution order
PREP -> CTRL (fixed order corpus, model, replicate, fold, cell; DONE markers; resumable) ->
PROBECPU (CPU, parallel) -> MECH2X2 -> HPO -> timing probe -> tag-2 -> FT -> MECHID ->
ledger lock -> STAT (rolling 1 %+1 % bitwise retrain checks; one-shot scoring; verifier
replay; number-checked insert). The audit runs on CPU/human time in parallel.

## 8. Audit protocol
`ser_v2/audit/protocol.py` (eligibility codes, lineage rule, endpoint dictionary Y_test/Y_val/N/U,
descriptor columns, caps), `sample.py` (beacon priority order; sequential screening to fixed
n = 60; time never a stopping rule), `adjudicate.py` (scripted adjudication; kappa reported,
never a gate; execution tier statistics restricted to the eligible subset; Wilson, partial
identification, exact hypergeometric frame envelope, U bounds). Estimation only: no audit
hypothesis test.

## 9. Deviations
Append-only `deviations.jsonl`; every deviation is cited by entry number in the paper.
