# SER v2 scoring contract (frozen at tag-1)

This document is the complete specification of (a) the files a run produces, (b) every
number the scorer reports, and (c) how the independent verifier must recompute them.
The scorer (`ser_v2/score.py`) and the verifier (`ser_v2/verify.py`) are written from this
document by different authors, share no code, and must agree to an absolute tolerance of
1e-12 on every floating value they both report.

## 1. Inputs

* `plan/run_plan.csv` — one row per unit. Columns: `unit_id, arm, corpus_level, base_corpus,
  panel_draw, model, cell, fold, r, seed_index, train_seed, config_sha256, split_sha256,
  n_fit, n_val, n_test, est_gpu_sec, cap_group, conditional, truncation_rank, status`.
  `corpus_level` carries the panel draw suffix for drawn panels (`cremad_24_d3`).
  For HPO units the config's grid index is inside the unit config (`hpo_config_index`).
* `plan/split_index.json` — `{key: {sha256, path, n_folds}}` for every split table.
* `plan/unit_configs.json` — `{config_sha256: config}`; HPO configs carry `hpo_config_index` (0..7).
* `plan/splits/<key>.json` — `{key, population:[relative_path...], folds:[{fold, test:[...],
  val:[...], fit?:[...], meta}]}` (paths, not indices).
* `manifests/<base>_manifest.csv` — columns `sample_index, corpus, relative_path, bytes, sha256,
  speaker, sex, label, label_index, sentence, take, intensity` (after hygiene; `plan/hygiene_log.json`).
  `n_classes` of a level = number of distinct `label_index` values of its base corpus manifest.
* `runs/<run_id>/units/<unit_id>/predictions.csv` — the outer-test predictions of that unit:
  `sample_index, relative_path, speaker, y_true, y_pred, logit_0 .. logit_{K-1}` (K = n_classes).
* `runs/<run_id>/units/<unit_id>/unit.json` — `{unit_id, arm, corpus_level, model, cell, fold, r,
  seed_index, train_seed, config, split_sha256, predictions_sha256, n_fit, n_val, n_test,
  best_epoch, epochs_run, val_uar_best, val_loss_best, gpu_seconds, started_at, finished_at,
  resumed, engine_version, status}`; `val_uar_best` is in percent.
* `runs/<run_id>/units/<unit_id>/DONE` — text file whose content is `predictions_sha256`.
* `runs/<run_id>/ledger.jsonl` — append-only attempt records (not scored; integrity only).
* `runs/<run_id>/deviations.jsonl` — append-only; entries cited by number in verdicts.
* `registry/hypothesis_registry.csv`, `registry/claim_map.json` — families, claims, rules.

`predictions_sha256` = SHA-256 of the exact bytes of `predictions.csv`.

## 2. Integrity checks (verifier must perform all; any failure is reported, never patched)

I1 every unit with status `done` has `DONE` == `unit.json.predictions_sha256` == SHA-256(predictions.csv).
I2 `unit.json.split_sha256` equals the plan row's `split_sha256`, and the set of `relative_path`
   in predictions.csv equals the `test` list of that unit's fold in the split table.
   I2b for every split table referenced by a done unit, SHA-256 of the exact bytes of
   `plan/splits/<key>.json` equals `plan/split_index.json[key].sha256` (reported under I2).
I3 `y_true` equals the manifest `label_index` for every row; `speaker` equals the manifest speaker.
   I3b every path in a referenced split table's `population` exists in the hygiene-filtered manifest
   of the unit's base corpus (reported under I3).
I4 no `relative_path` appears twice in one predictions.csv; `y_pred == argmax(logits)` row-wise
   (ties broken by lowest index).
I5 a *cell-run* (Section 3) is COMPLETE only if the union of its folds' prediction paths equals,
   exactly once, the union of the `test` lists of ALL folds of its split table (the table's *test
   population*). For CTRL/FT/HPO/PROBECPU tables that union equals `population`; for MECH2X2
   tables it is the checkerboard test set, a strict subset of `population`. Otherwise the
   cell-run is VOID and every contrast using it is voided.
I6 for HPO cell-runs, every fold has all 8 configs done, else the cell-run is VOID.
I7 `unit.json.config` equals `plan/unit_configs.json[plan row config_sha256]` (canonical JSON equality).

## 3. Cell-runs and per-speaker values

A **cell-run** is identified by `(arm, corpus_level, model, cell, r, seed_index)` plus, for HPO,
the selection rule below. Its OOF prediction set is the concatenation of its folds' predictions.

* **Main replicate units**: for arms CTRL, MECH2X2 and HPO those with `seed_index == r`
  (crossing units with `seed_index != r` are used only for D14); for arms FT and PROBECPU every
  unit is a main unit (`r` is always 0 there and `seed_index` indexes seeds / subsamples).
* **Per-speaker UAR** `U(s)` of a cell-run: for speaker `s`, over `s`'s OOF rows, recall_c =
  (#rows with y_true==c and y_pred==c)/(#rows with y_true==c) for every class c PRESENT among
  `s`'s rows; `U(s) = 100 * mean_c recall_c`. (Classes absent for that speaker are skipped.)
* **Replicate average**: `V[level, model, cell](s) = mean over the main cell-runs (r, or seed_index
  for FT/PROBECPU) of U(s)` over the cell-runs that are COMPLETE. If a level has no complete replicate for a model x cell,
  the value is undefined and dependent contrasts are `not tested`.
* **Model groups**: `scratch` = {cnn, resnet_se, transformer} on family levels and
  {cnn, resnet_se} on secondary levels; `probe` = {hubert_base, wavlm_base_plus, wav2vec2_base};
  `ft` = {wavlm_base_plus_ft}; `frozen_sr` = {wavlm_base_plus_frozen_sr}; `ridge` = the three
  `ridge_a1_<encoder>` models of arm PROBECPU. Group value `V[level, group, cell](s)` = mean over
  the group's models of `V[level, model, cell](s)` (all members must be defined).
* **FT seeds**: for `ft` and `frozen_sr`, replicate averaging is over `seed_index` values present
  for BOTH models on that level (r is always 0).
* **HPO selection** (arm HPO, cells `GR_hpo`, `GG_hpo`, `RR_hpo`): per fold, the selected config
  is the one with the largest `val_uar_best`; ties -> smallest `hpo_config_index`. The cell-run's
  OOF is the selected configs' test predictions. `val_sel(f)` = the selected config's
  `val_uar_best`. `test_max(f)` = max over configs of the fold-level UAR on the fold's test rows
  (UAR over classes present in the fold's test rows, in percent); `test_sel(f)` = the selected
  config's fold-level test UAR.
* **Mechanism block** (arm MECH2X2): cells are the conditions `none, spk, prm, both, both_sib,
  spk_half_h1, spk_half_h2`. Per-speaker UAR uses the speaker's test rows (each speaker is a test
  speaker in exactly one fold). For a speaker `s` with test fold f: `exposed_half(s)` is `h1` if
  s ∈ `meta.exposed_speakers` of the `spk_half_h1` fold f, else `h2`; `unexposed_half(s)` is the
  other one.

## 4. Contrasts (all in UAR percentage points, unit = speaker)

`D(s)` is defined per hypothesis; `mean`, CI and p follow Section 5.

| id | D(s) |
|---|---|
| N01 | V[subesco_980, scratch, RR] − V[subesco_980, scratch, GG] |
| N02 | V[subesco_980, probe, RR] − V[subesco_980, probe, GG] |
| N03 | V[subesco_980, scratch, RG] − V[subesco_980, scratch, GG] |
| N04 | V[subesco_980, scratch, RR] − V[subesco_980, scratch, RG] |
| N05 | V[MECH cremad, scratch, prm] − V[MECH cremad, scratch, none] |
| N06 | V[MECH cremad, scratch, spk] − V[MECH cremad, scratch, none] |
| N07 | V[MECH subesco_full, scratch, prm] − V[MECH subesco_full, scratch, none] |
| N08 | V[MECH subesco_full, scratch, spk] − V[MECH subesco_full, scratch, none] |
| N09 | V[MECH subesco_full, scratch, both_sib] − V[MECH subesco_full, scratch, both] |
| N10 | P24(s) − P91(s), where for each draw k: P24_k(s) = V[cremad_24_dk, scratch, RR](s) − V[cremad_24_dk, scratch, GG](s) for speakers in draw k; P24(s) = mean over draws containing s; P91(s) = mean over k of V[cremad_91m_dk, scratch, RR](s) − V[cremad_91m_dk, scratch, GG](s). Speakers = those in ≥1 cremad_24 draw. |
| N11/N12/N13 | on level L ∈ {ravdess, cremad, subesco_980}: (V[L, ft, RR] − V[L, ft, GG]) − (V[L, frozen_sr, RR] − V[L, frozen_sr, GG]) |
| N14 | on cremad: [val_sel(f(s), GR_hpo) − U_GR_hpo(s)] − [val_sel(f(s), GG_hpo) − U_GG_hpo(s)], where f(s) is the fold in which s is a test speaker under the respective cell's outer split (the two cells share the same outer split G, r=0). |
| N15 | on ravdess: mean_enc[V[ravdess, ridge_enc, RO] − V[ravdess, ridge_enc, GO]] − mean_enc[V[ravdess, probe_enc, RG] − V[ravdess, probe_enc, GG]] (PROBECPU cells RO/GO = Ridge on the CTRL RR/GG outer partitions r=0..2, no inner split; `n_val` = 0) |
| R01..R06 | as N01/N03/N04 with level ravdess (R01-R03) and cremad (R04-R06) |
| R07/R08 | as N02 with level ravdess / cremad |
| R09/R10/R11 | V[L, ft, RR] − V[L, ft, GG] on ravdess / cremad / subesco_980 |
| T01 | on cremad: mean_enc[V[cremad, ridge_enc, LOSOSUB] − V[cremad, ridge_enc, G5]] (PROBECPU cells; G5 = the CTRL GG r=0 outer folds without inner split; LOSOSUB = size-matched LOSO, three subsamples as seed_index 0..2, averaged per speaker) |
| T02 | on MECH cremad: U_{unexposed_half(s)}(s) − U_none(s) (scratch group) |

Descriptors (verifier recomputes D01-D09; D10-D18 come from other tools and are replayed there):

* D01 `V[L, scratch, GR] − V[L, scratch, GG]` per family level; also probe group.
* D02 for each family level and each model separately: sign of mean(V[RR] − V[GG]) and the count of
  models with positive sign (`x/3` scratch, `x/3` probe) — reported as `positive_models/total`.
* D03 corpus differences of the scratch RR−GG mean: (ravdess − cremad) and (ravdess − subesco_980);
  CI by two-sample speaker bootstrap: independently resample speakers within each level (same
  RNG: `RandomState(20260903)`, draw the first level's index matrix then the second's, both of
  shape (10000, n_level)), difference of means, percentile 2.5/97.5. Band ±3 pp: label `larger`
  only if the whole CI lies outside [−3, 3].
* D04 sub_1400_one vs sub_700_one and sub_700_two vs sub_700_one, scratch RR−GG per speaker,
  paired by speaker.
* D05 subesco_full: V[subesco_full, cnn, RR] − V[subesco_full, cnn, TG].
* D06 MECH per level (scratch): both − none; interaction = (both−none) − (spk−none) − (prm−none);
  G1(s) = U_spk(s) − U_none(s); G05(s) = U_{exposed_half(s)}(s) − U_none(s); crowding = G1 − G05;
  share = mean(G05)/mean(G1) (reported only when mean(G1) > 0).
* D07 per draw k: mean over speakers of P24_k and of P91_k; SD over draws.
* D08 RR_hpo: V[cremad, resnet_se, RR_hpo] − V[cremad, resnet_se, GG_hpo] per speaker; selected
  config indices per fold and cell; test-selection optimism per fold: `test_max(f) − test_sel(f)`
  for each cell, mean over folds.
* D09 honest DiD per speaker: (U_GR_hpo(s) − U_GG_hpo(s)) − (V[cremad, resnet_se, GR](s) −
  V[cremad, resnet_se, GG](s)) using the CTRL r=0 replicate only for the fixed-config term.
* D14 (crossing) for ravdess and subesco_980, cnn: the 3x3 matrix M[r][seed] of the mean over
  speakers of (U_RR − U_GG) using unit (r, seed_index=seed); row means, column means, grand mean;
  `sd_draw` = sample SD of row means, `sd_seed` = sample SD of column means.

## 5. Statistics (exact procedures)

* `mean` = arithmetic mean of D(s) over the speakers for which D is defined; `n` = count.
* Bootstrap CI: `rng = numpy.random.RandomState(20260903)`;
  `idx = rng.randint(0, n, size=(10000, n))`; `means = D[idx].mean(axis=1)`;
  `lo, hi = numpy.percentile(means, [2.5, 97.5], method="linear")`. A fresh RandomState is created
  for every hypothesis/descriptor (same seed), in the order the speakers are sorted by speaker id.
* Tie determinism: immediately before the Wilcoxon test, the sign test and the `n_nonzero` count,
  the difference vector is rounded, `D_r = numpy.round(D, 10)` (mean, median, sd and the CI use the
  unrounded D). This makes tie detection independent of floating-point summation order.
* Wilcoxon: `scipy.stats.wilcoxon(D_r, zero_method="wilcox", alternative="two-sided")` applied to the
  non-zero rounded differences (D_r[D_r != 0]); if none are non-zero, p = 1.0. `n_nonzero` reported.
* Sign test: when `n_zero > 0.2 n`: `scipy.stats.binomtest(n_pos, n_nonzero, 0.5, alternative="two-sided").pvalue`
  reported as `sign_test_p`; the family test still uses the Wilcoxon p (sign test is a sensitivity).
* Families are read from `hypothesis_registry.csv` (`family` column; claim-wise families such as
  N-decomp, N-mech, N-modern, N-supp, R, T). Only rows with `a_priori_status == confirmatory` enter
  their family; `m` = that family's confirmatory count (`family_size` column). Rows with
  `a_priori_status == estimate` are computed and reported exactly like the others (n, mean, CI,
  p) but receive `p_holm = null`, `reject = false` and verdict `estimate (not tested)` (or
  `not tested` when they cannot be computed).
* Holm within a family: sort p ascending; adjusted p_(i) = max over j ≤ i of min(1, (m − j + 1) p_(j));
  reject while adjusted ≤ 0.05 in step-down order (a non-rejection stops all later rejections).
  Missing confirmatory hypotheses enter with p = 1.0 and are labelled `not tested`.
* TOST (family T, confirmatory members only): margin from the registry; `se = sd(D, ddof=1)/sqrt(n)`;
  `p_low = 1 − t.cdf((mean + margin)/se, n−1)`, `p_high = t.cdf((mean − margin)/se, n−1)`;
  `p = max(p_low, p_high)`; the two T hypotheses are Holm-adjusted together at alpha 0.05;
  `ci90 = t.interval(0.90, n−1, loc=mean, scale=se)`.
* Directional reading: a Family N/R hypothesis with direction `>0` is `supported` iff its Holm
  rejection holds AND mean > 0; `not supported` if the test ran and either condition fails;
  `not tested` if voided/truncated. N15 has no direction: supported iff Holm rejects.
* Claim verdicts follow `registry/claim_map.json` structurally for every non-estimation claim,
  including type `replication` (`requires` = all listed supported; `requires_any` = at least one
  supported and none `not supported` with opposite-sign mean; `requires_k_of` = at least k
  supported); the `evidence` dict carries the per-hypothesis verdicts. If a required hypothesis is
  `not tested` and the rule cannot be satisfied by the remaining ones, the claim is `not tested`;
  the suffix ` (truncated, deviation entry N)` is appended (to claims and hypotheses) only when
  `runs/<run_id>/deviations.jsonl` contains an entry whose `hypotheses` list names the hypothesis
  (N = the entry's `entry` field, else its 1-based line number).

## 6. Output (`runs/<run_id>/results.json`)

```
{
  "schema": "ser-v2-results-1",
  "run_id": ..., "plan_sha256": ..., "registry_sha256": ..., "n_units_done": ..., "n_units_void": ...,
  "cell_runs": {"<arm>|<level>|<model>|<cell>|r<r>|s<seed>": {"complete": bool, "n_folds": int, "n_rows": int,
                 "speaker_uar": {"<speaker>": float}}},
  "hypotheses": {"N01": {"n": int, "n_nonzero": int, "mean": float, "median": float, "sd": float,
                         "ci_low": float, "ci_high": float, "wilcoxon_p": float, "sign_test_p": float|null,
                         "p_holm": float, "reject": bool, "verdict": "supported|not supported|not tested",
                         "family": "N", "m": 15, "speakers": ["..."], "d": [floats in speaker order]}},
  "tost": {"T01": {..., "p_low", "p_high", "p", "p_holm", "ci90": [lo, hi], "se", "equivalent": bool, "verdict"}}
           (a not-tested T row has p = 1.0 and p_low = p_high = ci90 = null),
  "claims": {"C2a": {"verdict": "...", "rule": "...", "evidence": {"N01": "supported"}}},
  "descriptors": {"D01": {...}, ...},
  "integrity": {"I1": {"pass": bool, "failures": [...]}, ...}
}
```
Float values are written with `repr` precision (no rounding). `numeric_insert.tex` is generated
from `results.json` by the scorer only; the verifier checks that every macro value in the tex
equals the corresponding results.json value after the stated rounding (2 decimals for UAR
points, 3 for p unless < 0.001 which prints as `<0.001`).

## 7. Verifier output

`runs/<run_id>/verification.json`: for every hypothesis / tost / claim / descriptor / integrity
item: `{"scorer": value, "verifier": value, "abs_diff": float, "pass": bool}`; a top-level
`pass` that is true only if every item passes; `max_abs_diff`. Mismatches are reported verbatim
and never corrected in results.json.
