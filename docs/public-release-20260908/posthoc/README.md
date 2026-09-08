# Post hoc sensitivity analysis of the completed 720-fit supplement

This is a public projection of the numerical follow-up completed after the primary results and review were available. It adds no training, no patience simulation and no new primary test. The original 24-draw ten-test family remains unchanged.

The [script](replay_posthoc.py) reads the public complete result tables, verifies their original manifest, reproduces all ten primary outcomes, and then calculates all 24 leave-one-draw-out versions of the whole Holm-10 family, 40 checkpoint-selection descriptions and descriptive correlations. Its public adaptation only removes the dependency on private reviewer text; the mathematical functions and scientific input checks are unchanged. See [source note](SOURCE_NOTE.json) and [complete result](results.json). All five scientific output blocks match the earlier local diagnostic exactly.

```text
python docs/public-release-20260908/posthoc/replay_posthoc.py --repo . --out /new/path/posthoc.json
```

Use a new output path outside the repository. Dependencies are the same NumPy/SciPy stack as the small table replay. Do not run with Python optimization flags that disable assertions.

| Original endpoint | Rejects in full 24-draw family? | Rejects after leave-one-draw-out, out of 24 |
|---|---|---:|
| HuBERT CREMA-D D_CE | Yes | 24 |
| HuBERT CREMA-D J | Yes | 24 |
| HuBERT SUBESCO D_CE | Yes | 24 |
| HuBERT SUBESCO J | Yes | 24 |
| HuBERT RAVDESS D_CE | Yes | 15 |
| HuBERT RAVDESS J | No | 0 |
| WavLM SUBESCO L_CE | No | 0 |
| WavLM SUBESCO L_J | No | 0 |
| WavLM RAVDESS L_CE | Yes | 24 |
| WavLM RAVDESS L_J | No | 4 |

The HuBERT RAVDESS CE contrast is sensitive to the significance threshold under this diagnostic, although every leave-one-out mean remains positive. The converse sensitivity of the originally non-rejected RAVDESS window interaction is also retained. No draw is removed from the primary analysis.

In SUBESCO, the unseen-CE selected epoch and outer score are unchanged for 120/120 contexts between the two windows, while the seen-CE epoch changes in 15/120. The overall L_CE is therefore small and nonzero; it is not a structural zero of the complete contrast. Correlation participation ratios are descriptive properties of these matrices, not counts of independent scientific discoveries.

These diagnostics were performed after seeing results. They do not establish a new confirmatory replication, justify changing the ten-test family, or identify a universally optimal training length.
