# Study II: complete verified estimates

Pointwise 95% conditional bootstrap intervals; n = 91 speakers. Draws are not independent samples.

All planned model/budget/condition combinations are retained. No p-values or simultaneous-coverage claims.
CSV files preserve the supplied numerical precision; Markdown displays two decimal places.

| Kind | Model | B | Allocation / contrast | Unit | Estimate [95% interval] |
|---|---|---:|---|---|---|
| absolute | Ridge–WavLM | 288 | S12 / seen | UAR % | 44.74 [43.45, 45.99] |
| absolute | Ridge–WavLM | 288 | S12 / new | UAR % | 28.54 [27.75, 29.30] |
| absolute | Ridge–WavLM | 288 | S48 / seen | UAR % | 44.54 [43.20, 45.86] |
| absolute | Ridge–WavLM | 288 | S48 / new | UAR % | 29.27 [28.57, 30.00] |
| absolute | Ridge–WavLM | 576 | S12 / seen | UAR % | 46.96 [45.51, 48.41] |
| absolute | Ridge–WavLM | 576 | S12 / new | UAR % | 28.33 [27.67, 28.99] |
| absolute | Ridge–WavLM | 576 | S48 / seen | UAR % | 46.50 [45.12, 47.84] |
| absolute | Ridge–WavLM | 576 | S48 / new | UAR % | 28.39 [27.69, 29.08] |
| absolute | CNN | 288 | S12 / seen | UAR % | 41.27 [39.73, 42.83] |
| absolute | CNN | 288 | S12 / new | UAR % | 40.24 [38.69, 41.83] |
| absolute | CNN | 288 | S48 / seen | UAR % | 42.57 [40.85, 44.29] |
| absolute | CNN | 288 | S48 / new | UAR % | 41.32 [39.66, 42.96] |
| absolute | CNN | 576 | S12 / seen | UAR % | 43.14 [41.48, 44.80] |
| absolute | CNN | 576 | S12 / new | UAR % | 41.06 [39.42, 42.67] |
| absolute | CNN | 576 | S48 / seen | UAR % | 45.29 [43.55, 47.07] |
| absolute | CNN | 576 | S48 / new | UAR % | 43.22 [41.55, 44.87] |
| contrast | Ridge–WavLM | 288 | new (primary): S48−S12 | percentage points | 0.73 [0.07, 1.43] |
| contrast | Ridge–WavLM | 288 | seen: S48−S12 | percentage points | -0.20 [-1.05, 0.62] |
| contrast | Ridge–WavLM | 288 | (new S48−S12) − (seen S48−S12) | percentage points | 0.94 [-0.09, 1.98] |
| contrast | Ridge–WavLM | 576 | new (primary): S48−S12 | percentage points | 0.06 [-0.64, 0.76] |
| contrast | Ridge–WavLM | 576 | seen: S48−S12 | percentage points | -0.46 [-1.40, 0.44] |
| contrast | Ridge–WavLM | 576 | (new S48−S12) − (seen S48−S12) | percentage points | 0.53 [-0.55, 1.62] |
| contrast | CNN | 288 | new (primary): S48−S12 | percentage points | 1.08 [0.50, 1.67] |
| contrast | CNN | 288 | seen: S48−S12 | percentage points | 1.30 [0.70, 1.91] |
| contrast | CNN | 288 | (new S48−S12) − (seen S48−S12) | percentage points | -0.22 [-0.92, 0.49] |
| contrast | CNN | 576 | new (primary): S48−S12 | percentage points | 2.16 [1.54, 2.78] |
| contrast | CNN | 576 | seen: S48−S12 | percentage points | 2.15 [1.57, 2.72] |
| contrast | CNN | 576 | (new S48−S12) − (seen S48−S12) | percentage points | 0.01 [-0.60, 0.65] |

## Complete-draw sensitivity

These three draws reuse the same 91 speakers.

| Kind | Model | B | Allocation / contrast | Unit | Draw 0 | Draw 1 | Draw 2 | Range |
|---|---|---:|---|---|---:|---:|---:|---|
| absolute | Ridge–WavLM | 288 | S12 / seen | UAR % | 45.20 | 44.55 | 44.48 | [44.48, 45.20] |
| absolute | Ridge–WavLM | 288 | S12 / new | UAR % | 29.45 | 27.72 | 28.44 | [27.72, 29.45] |
| absolute | Ridge–WavLM | 288 | S48 / seen | UAR % | 44.47 | 43.76 | 45.39 | [43.76, 45.39] |
| absolute | Ridge–WavLM | 288 | S48 / new | UAR % | 30.15 | 29.76 | 27.90 | [27.90, 30.15] |
| absolute | Ridge–WavLM | 576 | S12 / seen | UAR % | 46.82 | 47.64 | 46.44 | [46.44, 47.64] |
| absolute | Ridge–WavLM | 576 | S12 / new | UAR % | 28.03 | 27.94 | 29.01 | [27.94, 29.01] |
| absolute | Ridge–WavLM | 576 | S48 / seen | UAR % | 46.00 | 46.48 | 47.03 | [46.00, 47.03] |
| absolute | Ridge–WavLM | 576 | S48 / new | UAR % | 27.55 | 28.85 | 28.77 | [27.55, 28.85] |
| absolute | CNN | 288 | S12 / seen | UAR % | 41.31 | 41.74 | 40.78 | [40.78, 41.74] |
| absolute | CNN | 288 | S12 / new | UAR % | 40.84 | 39.91 | 39.97 | [39.91, 40.84] |
| absolute | CNN | 288 | S48 / seen | UAR % | 42.67 | 42.77 | 42.28 | [42.28, 42.77] |
| absolute | CNN | 288 | S48 / new | UAR % | 41.65 | 40.89 | 41.42 | [40.89, 41.65] |
| absolute | CNN | 576 | S12 / seen | UAR % | 43.27 | 42.49 | 43.66 | [42.49, 43.66] |
| absolute | CNN | 576 | S12 / new | UAR % | 41.46 | 40.55 | 41.17 | [40.55, 41.46] |
| absolute | CNN | 576 | S48 / seen | UAR % | 45.15 | 45.30 | 45.42 | [45.15, 45.42] |
| absolute | CNN | 576 | S48 / new | UAR % | 43.32 | 43.10 | 43.24 | [43.10, 43.32] |
| contrast | Ridge–WavLM | 288 | new (primary): S48−S12 | percentage points | 0.70 | 2.04 | -0.54 | [-0.54, 2.04] |
| contrast | Ridge–WavLM | 288 | seen: S48−S12 | percentage points | -0.74 | -0.79 | 0.92 | [-0.79, 0.92] |
| contrast | Ridge–WavLM | 288 | (new S48−S12) − (seen S48−S12) | percentage points | 1.44 | 2.83 | -1.46 | [-1.46, 2.83] |
| contrast | Ridge–WavLM | 576 | new (primary): S48−S12 | percentage points | -0.48 | 0.91 | -0.24 | [-0.48, 0.91] |
| contrast | Ridge–WavLM | 576 | seen: S48−S12 | percentage points | -0.82 | -1.16 | 0.59 | [-1.16, 0.59] |
| contrast | Ridge–WavLM | 576 | (new S48−S12) − (seen S48−S12) | percentage points | 0.34 | 2.07 | -0.84 | [-0.84, 2.07] |
| contrast | CNN | 288 | new (primary): S48−S12 | percentage points | 0.81 | 0.98 | 1.45 | [0.81, 1.45] |
| contrast | CNN | 288 | seen: S48−S12 | percentage points | 1.36 | 1.03 | 1.51 | [1.03, 1.51] |
| contrast | CNN | 288 | (new S48−S12) − (seen S48−S12) | percentage points | -0.55 | -0.06 | -0.06 | [-0.55, -0.06] |
| contrast | CNN | 576 | new (primary): S48−S12 | percentage points | 1.86 | 2.56 | 2.07 | [1.86, 2.56] |
| contrast | CNN | 576 | seen: S48−S12 | percentage points | 1.88 | 2.80 | 1.76 | [1.76, 2.80] |
| contrast | CNN | 576 | (new S48−S12) − (seen S48−S12) | percentage points | -0.02 | -0.25 | 0.31 | [-0.25, 0.31] |
