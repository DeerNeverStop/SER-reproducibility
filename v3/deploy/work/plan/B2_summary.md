# B2 plan summary (SER26-DEPLOY-2, spec DEPLOY-2)

generated_at: 2026-09-05T18:57:21.185872+00:00
n_units: 390

## Units per model

| model | engine | units |
|---|---|---|
| cnn | p1_frozen | 90 |
| ridge_hubert_base | ridge | 90 |
| ridge_wav2vec2_base | ridge | 90 |
| ridge_wavlm_base_plus | ridge | 90 |
| wavlm_ft | wavlm_partial_ft | 30 |

## Units per level x model

| level | cnn | ridge_hubert_base | ridge_wav2vec2_base | ridge_wavlm_base_plus | wavlm_ft |
|---|---|---|---|---|---|
| ravdess | 30 | 30 | 30 | 30 | 10 |
| cremad | 30 | 30 | 30 | 30 | 10 |
| subesco_980 | 30 | 30 | 30 | 30 | 10 |

## Notes

- seed rule: cnn: train_seed = v2 CTRL unit (same level, cell, r, fold, seed_index) when present, else stable_u32('SER26|train|<level>|<fold>|<seed_index>') (identical to v2 seeds.crossing_train_seed; coincides with every present v2 unit); wavlm_ft: v2 FT GG unit of the same level/fold, seed_index 0 (= stable_u32('SER26|train|<level>_ft|<fold>|0')), used for both cells; ridge: 0 (deterministic solver).
- cnn units with derived train_seed: 20
- feature caches: v2 base_corpus caches (subesco_980 level uses the subesco_980__* caches, as v2 did); labels/speakers from the v2 base manifest (common.LEVELS).
- output contract: see the docstring of v3/deploy/calib.py or plan['contract'].
