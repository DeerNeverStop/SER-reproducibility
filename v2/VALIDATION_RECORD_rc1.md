# Validation record — v2 preregistration candidate rc1 (2026-09-03)

Everything below was executed in a CPU-only session without corpus audio. Numbers from
synthetic fixtures are mechanics checks, never scientific results.

| Gate | Evidence | Result |
|---|---|---|
| G1 registry validates | `python -c "from ser_v2 import registry; registry.export(...)"` | 28 hypotheses, all bound to claims; families N-decomp 4, N-mech 4 (+2 estimates), N-modern 0 (+3 estimates), N-supp 2, R 11, T 2 |
| G2 split assertions on the real manifests | `python -m ser_v2.run_plan --manifests manifests --out plan_rc1 --p1-splits ...` | 4,943 units; deterministic (identical `run_plan_sha256` on rebuild); hygiene log: CREMA-D 7,442→7,435, RAVDESS 1,440→1,439, SUBESCO-980 980→980 |
| G3a scorer vs independent verifier | full synthetic run (4,943 units, plan_cmp = plan_rc1 + placeholder SUBESCO manifest) | `evidence/rc1/verification_summary_synthetic_full.json`: **pass = true, 1,320 items, 0 failed, max_abs_diff 2.8e-14**, verifier integrity pass |
| G3b mutation battery | `python -m ser_v2.mutations ...` on the small run and on the full synthetic run | small run: 16/17 (3 cases skipped for absent arms; `evidence/rc1/mutation_report_small.json`); **full run: 18/18 behaved as specified** (`evidence/rc1/mutation_report_full.json`; two family-size expectations were updated from the pre-restructuring constant to the registry value and rerun); world-level cases: no false support in a null world, sign flip read as `not supported` |
| G3c unit tests | `python -m pytest tests -q` | 21 passed |
| G4 runner with the frozen P1 engine | CPU dry run, unit `8b66ec62…` (ravdess, cnn, RR, fold 0, r 0) on synthetic log-mel | `evidence/rc1/dry_run_p1_cnn_unit.json`: 863 fit / 288 val / 288 test rows, 100 epochs, checkpoint persisted, DONE == predictions sha; synthetic engine, linear probe and Ridge engines exercised through the fixture/scorer path; WavLM engines and SSL caches need torchaudio + audio (author machine) |
| Audit statistics | `ser_v2/audit` self-test | frame envelope reproduces the 2026 paper (N=286, n=60, Y=44, U=7 → K=[177,263], 61.9–92.0 %) |
| Power table | `python -m ser_v2.power_sim --reps 1000` | `registry/power_table.json`; a priori demotions: N07, N08, N11, N12, N13 |

Pinned hashes of the frozen files: `PINS_rc1.json`.

Interpretations the verifier author recorded (all now written into `SPEC_SCORING_CONTRACT.md`):
tie determinism by rounding D to 10 decimals before rank tests; I5 uses the split table's test
population; per-replicate half-exposure bookkeeping; structural claim rules for replication
claims; truncation suffix only with a `deviations.jsonl` entry naming the hypothesis.

Known limits of rc1: the SUBESCO full-corpus manifest is a synthetic placeholder (its split
tables and unit ids will change when the author supplies the real file; SUBESCO-980 is pinned);
`plan_rc1/splits/` (63 MB) is not committed — it is regenerated deterministically and pinned by
`split_index.json`; WavLM fine-tuning engines are untested here.
