# Tag-2 record — off-family timing probe and derived FT cap (2026-09-03)

Recorded after tag-1 and before any FT, HPO or MECHID unit. This file adds nothing but timing; the
scientific configuration, registry, plan and verifier are exactly those frozen at tag-1.

Probe: the CREMA-D FT cell (RR, fold 0, seed 0) run end to end with the real engines on SYNTHETIC
audio at CREMA-D cardinality and per-file durations (`tools/timing_probe.sh`; audio in
`E:\claudework_data\ICASSP2027-corpora\synthetic_cremad_timing_probe`, generated from the manifest's
byte sizes). Outcome-blind by construction. Evidence: `evidence/rc2/timing_probe/`.

| Engine | n_fit / n_val / n_test | epochs run | GPU s | s / epoch |
|---|---|---|---|---|
| wavlm_partial_ft (top 4 layers + pool + head, fp16, batch 16, 3 s crops) | 4461 / 1487 / 1487 | 15 | 312.3 | 20.8 |
| wavlm_frozen_same_regime (pool + head) | 4461 / 1487 / 1487 | 15 | 236.0 | 15.7 |

Derived planning values (15-epoch worst case, scaled by fit+val size, +60 s overhead per unit):

| Level | wavlm_base_plus_ft min/fold | wavlm_base_plus_frozen_sr min/fold |
|---|---|---|
| ravdess | 2.0 | 1.8 |
| cremad | 6.2 | 4.9 |
| subesco_980 | 1.7 | 1.5 |

Projected FT arm: fine-tune 2.27 h + frozen comparator 1.92 h = 4.18 h unconditional;
conditional CREMA-D seed 1 adds 1.86 h. Both are far below the frozen FT cap (26 h)
and the program cap (50 h), so **the conditional units run and no preregistered truncation is triggered**.
The planning constants in `registry/arms.json` (`ft_fold_min_planning` 12 / 36 / 9 min) are superseded by this
record for scheduling only; the registry file itself is unchanged.
