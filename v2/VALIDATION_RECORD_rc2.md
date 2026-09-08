# Validation record — v2 preregistration rc2 (2026-09-03, author's machine)

rc2 = rc1 + the real SUBESCO corpus + real feature caches + four code fixes found while running the
PREP arm on the training machine (Windows 11, RTX 5070 12 GB, `ser_gpu` env: torch 2.11.0+cu128,
torchaudio 2.11.0+cu128, librosa 0.11.0, scikit-learn 1.9.0, scipy 1.18.0). Numbers from synthetic
fixtures are mechanics checks, never scientific results. No real-data unit was fitted before tag-1.

## rc1 -> rc2 changes (all before tag-1)

| # | File | Change | Why |
|---|---|---|---|
| 1 | `manifests/subesco_manifest.csv` (new), `manifests/subesco_980_manifest.csv` | Real SUBESCO byte manifest built from `E:\claudework_data\ICASSP2027-corpora\subesco\extracted` (7,000 wav, 20 speakers, 10 sentences, 7 emotions; 2 exact-duplicate groups -> H2 keeps the lexically first path -> 6,998). The 980 panel manifest was rebuilt from it: same path order, same speaker/label/sentence/take/sha256 as the rc1 pinned list, real byte sizes instead of 0. All 980 pinned SHA-256 match the audio on disk. | rc1 had a synthetic placeholder for the full corpus |
| 2 | `ser_v2/run_plan.py` | `split_index.json` path field written as POSIX (`as_posix()`) | on Windows the field contained backslashes, which changed the pinned hash of `split_index.json` although every split table SHA-256 was identical; the fixed code reproduces `PINS_rc1.json` bit-for-bit |
| 3 | `ser_v2/features.py` | SSL post-projection state taken from `model.encoder.feature_projection(...)` (shape `[1, T, 768]`) | torchaudio's `Wav2Vec2Model` keeps `feature_projection` under `encoder`; rc1 was written blind (no torchaudio) and both the attribute path and an extra `[0]` index were wrong |
| 4 | `ser_v2/train.py` | per-arm cap check skipped when `cap == 0` (CPU arms) | the runner refused every PROBECPU unit ("cap reached 0.00 h of 0.0") |
| 5 | `tools/` (new, outside the frozen scorer) | `mechid.py` (D10/D11), `ridge_sweep.py` (D12/D13), `status.py`, `make_pins.py`, `run_main_gpu.sh`, `run_main_cpu.sh`, `timing_probe.sh` | descriptor tools the rc1 package listed as "separate tools" but did not ship |

Registry files (`registry/*`), `SPEC_SCORING_CONTRACT.md`, `score.py`, `verify.py`, `splits.py`,
`seeds.py`, `stats.py` are byte-identical to `PINS_rc1.json`.

## Gates

| Gate | Evidence | Result |
|---|---|---|
| G1 registry validates | unchanged from rc1 (hashes in `PINS_rc2.json` equal `PINS_rc1.json`) | 28 hypotheses bound; families N-decomp 4, N-mech 4 (+2 est.), N-modern 0 (+3 est.), N-supp 2, R 11, T 2 |
| G2 split assertions on the real manifests | `python -m ser_v2.run_plan --manifests manifests --out plan_rc2 --p1-splits ../results/protocol_premium/splits` | 4,943 units, every population `real`; hygiene CREMA-D 7,442->7,435, RAVDESS 1,440->1,439, SUBESCO 7,000->6,998, SUBESCO-980 980->980; `run_plan_sha256 = 2acc6581…`; rc1 determinism re-checked on this machine (`runs/prep/plan_rc1_check`: run_plan/split_index/unit_configs/hygiene all equal `PINS_rc1.json`) |
| G3a scorer vs independent verifier | full synthetic run on `plan_rc2` (4,943 units, `runs/prep/syn_rc2`) | `evidence/rc2/verification_summary_synthetic_full.json`: **pass = true, 1,320 items, 0 failed, max_abs_diff 2.8e-14**; integrity I1–I7 pass |
| G3b mutation battery | `python -m ser_v2.mutations … --work runs/prep/mut_rc2` on the full synthetic run | **20/20 cases behaved as specified, 0 skipped** (`evidence/rc2/mutation_report_full.json`; 18 fault injections incl. M15 conditional truncation, M18 family placeholder, M20 hygiene tamper, M21 crossing isolation, M23 TOST missing, M24 logit column, M25 mech condition; world-level: null world gives no false support, sign flip reads as `not supported`) |
| G3c unit tests | `python -m pytest tests -q` in `ser_gpu` | 21 passed (40 s) |
| G4 runner with every engine | synthetic caches for all four bases (`runs/prep/syn_features`), `runs/prep/dryrun_rc2` | 9 units DONE, 0 failed: p1_frozen cnn (ravdess GG f0, 25.8 s), resnet_se (subesco_980 GG f0), transformer (cremad GG f0, 167 s), cnn TG (subesco_full), linear_probe (ravdess wavlm GG f0), MECH2X2 cnn (subesco_full `both` f0), HPO resnet_se (cremad GG_hpo f0), ridge_a1 (cremad G5, ravdess RO); all on CUDA except Ridge (CPU) |
| G4 WavLM engines (`wavlm_partial_ft`, `wavlm_frozen_same_regime`) | `tools/timing_probe.sh`: FT cell cremad RR fold 0 seed 0 on synthetic audio at CREMA-D cardinality and durations (`runs/prep/timing_probe`) | both engines ran end to end on CUDA, fp16 autocast: partial FT 15 epochs, 312 s (20.8 s/epoch, n_fit 4461); frozen comparator 15 epochs, 236 s (15.7 s/epoch); checkpoints persisted; DONE == predictions sha |
| Off-family timing probe (feeds tag-2 only) | same two units | see `TAG2_TIMING_PROBE.md`: projected FT arm 4.18 h (+1.86 h conditional) vs planned 17.33 (+8.0) h; no truncation triggered |
| Feature caches | `features/` (content-addressed; gitignored) | log-mel `[n, 64, 128]` float32 and 13-state SSL `[n, 13, 768]` float16 for ravdess (1,439), cremad (7,435), subesco (6,998), subesco_980 (980); all finite; SSL weights: torchaudio bundles HUBERT_BASE, WAVLM_BASE_PLUS, WAV2VEC2_BASE (weight SHA-256 in each cache's `.json`) |
| G5 audit frame / rater forms | not executed in this run | the audit arm (C1, D17) needs GitHub search quota and human raters; C1 is carried from the 2026 record-279 closure (N = 286, n = 60, Y = 44, U = 7) and is reported as such, never as a v2 result |

Pinned hashes of the frozen files: `PINS_rc2.json`.

Known limits of rc2: G5 not executed (above); `plan_rc2/splits/` (63 MB) is regenerated deterministically
and pinned by `split_index.json`; the timing probe used synthetic audio, so its wall-clock is an
engineering estimate, not a scientific number.
