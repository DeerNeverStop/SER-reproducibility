# Shared facts for the completed program

Plan SHA-256: `393109434af0bfb6d18205f8f3713aa5e08d08f0f0ac3ffc4a63f8e635422c25`

All numbers are copied from one result bound to complete 384-unit acceptance and an independent numerical audit. All prespecified comparisons are retained. JSON preserves source precision; tables display six decimals.

## Six prespecified tests

δCE = T(seen-CE) − T(unseen-CE); δUAR = T(seen-UAR) − T(unseen-UAR); J = δCE − δUAR. T is outer-test UAR and differences are percentage points (pp). δCE is neither a CE-loss difference nor V−T optimism.

| Corpus | Effect | Mean (pp) | Pointwise 95% t CI (pp) | Raw two-sided p | Six-test Holm p | Status |
|---|---|---:|---|---:|---:|---|
| CREMA-D | δCE | 0.935000 | [0.287333, 1.582667] | 0.00659747 | 0.0263899 | estimated |
| CREMA-D | J | 0.936625 | [0.210250, 1.663000] | 0.0137569 | 0.0412708 | estimated |
| SUBESCO | δCE | 2.247024 | [0.944793, 3.549254] | 0.00162708 | 0.00830927 | estimated |
| SUBESCO | J | 2.559524 | [1.103031, 4.016017] | 0.00138488 | 0.00830927 | estimated |
| RAVDESS | δCE | 0.486111 | [-0.197613, 1.169835] | 0.154905 | 0.309809 | estimated |
| RAVDESS | J | 0.468750 | [-0.632537, 1.570037] | 0.387697 | 0.387697 | estimated |

Each test uses 24 complete draws, df=23, after averaging five folds per draw. Intervals are pointwise 95% t intervals, not simultaneous intervals; Holm adjusts only the six p values. Undefined zero-variance tests and intervals remain NA and do not establish equivalence. Inference is conditional on the finite corpus and fixed program.

## Absolute outer UAR and δUAR (descriptive)

| Corpus (native classes) | seen-CE UAR (%) | unseen-CE UAR (%) | seen-UAR UAR (%) | unseen-UAR UAR (%) | last-15 UAR (%) | δUAR (pp) |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D (6) | 56.253268 | 55.318268 | 57.153571 | 57.155196 | 53.715202 | -0.001625 |
| SUBESCO (7) | 47.113095 | 44.866071 | 47.425595 | 47.738095 | 47.291667 | -0.312500 |
| RAVDESS (8) | 33.940972 | 33.454861 | 32.942708 | 32.925347 | 32.821181 | 0.017361 |

Each corpus has 120 main trajectories (24 draws × 5 folds). The four rules select checkpoints from the same trajectory; last is fixed at epoch 15. CI and p are NA (not tested) for this table. Native class sets/tasks differ, so absolute scores do not identify language effects or rank corpus difficulty fairly.

## Checkpoint epochs and oracle shortfall (descriptive)

| Corpus | Rule | Mean selected epoch | Epoch:count (total 120) | Mean oracle shortfall (pp) |
|---|---|---:|---|---:|
| CREMA-D | seen_ce | 6.708333 | 3:1; 4:14; 5:23; 6:24; 7:18; 8:19; 9:11; 10:4; 11:4; 12:1; 14:1 | 2.339008 |
| CREMA-D | unseen_ce | 5.891667 | 2:1; 3:7; 4:18; 5:28; 6:26; 7:17; 8:13; 9:7; 10:3 | 3.274009 |
| CREMA-D | seen_uar | 10.533333 | 5:2; 6:8; 7:12; 8:9; 9:14; 10:15; 11:12; 12:13; 13:12; 14:15; 15:8 | 1.438705 |
| CREMA-D | unseen_uar | 10.091667 | 4:2; 5:6; 6:6; 7:8; 8:16; 9:17; 10:12; 11:18; 12:6; 13:7; 14:11; 15:11 | 1.437081 |
| CREMA-D | last | 15.000000 | 15:120 | 4.877074 |
| SUBESCO | seen_ce | 10.708333 | 4:1; 6:2; 7:8; 8:13; 9:14; 10:27; 11:9; 12:16; 13:9; 14:10; 15:11 | 4.940476 |
| SUBESCO | unseen_ce | 7.916667 | 2:1; 3:4; 4:3; 5:10; 6:15; 7:22; 8:22; 9:11; 10:16; 11:5; 12:7; 13:1; 14:2; 15:1 | 7.187500 |
| SUBESCO | seen_uar | 12.541667 | 6:1; 7:1; 8:6; 9:3; 10:13; 11:8; 12:19; 13:21; 14:22; 15:26 | 4.627976 |
| SUBESCO | unseen_uar | 11.975000 | 6:3; 7:3; 8:6; 9:9; 10:11; 11:17; 12:12; 13:15; 14:27; 15:17 | 4.315476 |
| SUBESCO | last | 15.000000 | 15:120 | 4.761905 |
| RAVDESS | seen_ce | 14.350000 | 3:1; 4:1; 12:2; 13:5; 14:39; 15:72 | 2.430556 |
| RAVDESS | unseen_ce | 13.858333 | 3:2; 4:2; 5:1; 11:2; 12:7; 13:8; 14:36; 15:62 | 2.916667 |
| RAVDESS | seen_uar | 13.658333 | 2:1; 3:1; 9:1; 10:4; 11:5; 12:9; 13:12; 14:39; 15:48 | 3.428819 |
| RAVDESS | unseen_uar | 13.475000 | 2:1; 5:1; 9:3; 10:5; 11:4; 12:16; 13:9; 14:35; 15:46 | 3.446181 |
| RAVDESS | last | 15.000000 | 15:120 | 3.550347 |

Oracle shortfall is the trajectory's maximum outer UAR across all 15 epochs minus outer UAR at the rule's checkpoint. The oracle never selects a retained checkpoint. CI/p are NA; these descriptions do not authorize post hoc epoch or primary-endpoint changes.

## Other prespecified descriptions

| Corpus | Mean epochs 8–10 minus mean epochs 14–15 (pp) | CE-rule epoch agreement (fraction) | UAR-rule epoch agreement (fraction) |
|---|---:|---:|---:|
| CREMA-D | 0.207526 | 0.583333 | 0.375000 |
| SUBESCO | -2.894345 | 0.266667 | 0.275000 |
| RAVDESS | -8.977141 | 0.766667 | 0.425000 |

This table uses the prespecified middle/late summary of complete trajectories. Agreement is a fraction (0–1), not a percentage. CI/p are NA (not tested).

| Last-epoch role-swap control | Pairs | Additional B fits | Mean E (pp) | CI | p |
|---|---:|---:|---:|---|---|
| CREMA-D, fold 0, epoch 15 | 24 | 24 | 2.828414 | NA | NA |

E = 0.5 × [(QA(MA15) − QA(MB15)) + (QB(MB15) − QB(MA15))]. This describes 24 prespecified fold-0 pairs replacing whole training groups, not an individual speaker causal effect. Query data also serve checkpoint selection for main A trajectories; they are not globally unused for selection.
