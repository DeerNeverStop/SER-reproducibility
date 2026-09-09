# Post hoc selection-window figure

[window_selection.pdf](window_selection.pdf) is a native vector figure sized **178 × 58 mm**, with a [300 dpi preview](window_selection.png). Blue solid lines show CE-based selection; orange dashed lines show UAR-based selection. Both panels use the same vertical scale. The vertical guides mark selection windows 15 and 45, both evaluated on the current archived 45-epoch trajectories.

Only the archived [curves.csv](../../../docs/autodl-supplement-20260908/results/curves.csv) supplies checkpoint choices and outer UAR values. At each candidate window W, each rule selects the earliest validation optimum within epochs 1…W. The outer UAR difference is calculated within each fold, five folds are averaged within each draw, and the resulting 24 draw values are averaged. Shading is mean ± t(0.975,23) × SD(draw means)/√24. It is a **pointwise**, conditional interval, not a simultaneous confidence band or an added family of hypothesis tests. Continuous windows are post hoc descriptions; their shape does not establish a mechanism, threshold, or population saturation.

The long trajectories are the same at all windows. W15 is their current 15-epoch prefix, not a replay of the older independent 384-fit program. These profiles do not simulate training termination or estimate saved compute.

The [generation audit](window_selection_audit.json) binds input/source/output SHA and records endpoint checks: all **1,920** selected epochs and their outer values at W15/W45 match the frozen table; **960** unit differences, **192** draw values, and **24** mean/interval values also match, with maximum absolute difference **8.9e-15 pp**. The original scorer uses exact Fraction UAR for ties. Here the archived full-precision float serialization is used, and all W15/W45 choices are explicitly checked against the original selections. No low-precision rounding is introduced.

Numerical outputs are [4,320 draw rows](window_selection_draws.csv) and [180 summary rows](window_selection_summary.csv). No new p values are calculated. [Layout QA](window_selection_layout_qa.json) records one-page PDF dimensions, embedded Type0 fonts with ToUnicode, and an extracted minimum text size of **9.2 pt** at native size. Poppler's 300 dpi PDF render was visually inspected with no clipping or overlap. Final manuscript embedding must preserve this scale or undergo a separate size check.

## Reproduction

From the repository root, use Python with Matplotlib and SciPy:

```sh
python -B paper/review-revision-20260909/scripts/make_window_figure.py --repo . --out /path/to/new-empty-window-figure-directory
```

The output directory must be new or empty. The source tables are checked against the archived result's table hashes and W15/W45 numerical references; source and input bytes are checked again before the audit is written. Paths recorded in the artifact are repository-relative and contain no machine directories or private links. Exact rendered PDF bytes may depend on library/font versions; the generation audit records those versions. The generation tool does not load model weights or predictions, start training, or access the network.

## Manuscript use and caption

From the adjacent English source directory:

```tex
\includegraphics[width=\textwidth]{../figures/window_selection.pdf}
```

Suggested caption:

> **Post hoc selection-window profiles on archived WavLM 45-epoch trajectories.** At each window W, checkpoints are selected by seen or unseen validation CE/UAR, with earliest ties. Lines show the mean paired outer-UAR difference (seen minus unseen); shaded pointwise 95% t intervals use 24 draw means, each averaging five folds. Intervals are conditional on the fixed corpora, panels and training program, are not simultaneous bands, and do not add confirmatory tests. Vertical guides mark selection windows 15 and 45.
