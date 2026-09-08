# Speaker-coverage execution

The current protocol is [EXECUTION_PLAN.md](EXECUTION_PLAN.md). The original
[review packet](../../docs/research-plan-20260906/README.md) remains a historical
v2 proposal; its E3/G/E4 budget is not the current execution budget.

This branch contains a new planner, a separate runner and an independent
verifier. It reuses the existing corpus manifests, cached input features and
model architectures without rewriting older studies. Selection and layout are
fixed before new SER outcomes. All 720 formal fits (270 CNN, 270 Ridge and
180 WavLM fine-tuning) and 19 isolated technical pilot fits are complete.
The [final report](reports/RUN_REPORT.md) contains every prespecified contrast,
the independent numerical replay, measured resources and the stopped-cloud
receipt. The primary CNN best-checkpoint C−U effect was −0.109 percentage points
(conditional 95% interval −0.715 to +0.509); WavLM fine-tuning at its best
checkpoint also showed no clear benefit. These are bounded results, not an
equivalence or universal-ineffectiveness finding.

Review background is in [Claude's review](review/CLAUDE_REVIEW.md),
[our evidence audit](review/RESPONSE.md), and the time-stamped
[cloud-options check](review/CLOUD_OPTIONS.md). The cloud-options file describes
the state before the operator resumed a pod; execution receipts record the later
state and actual costs. No credential belongs in this package.

The generated plan, raw predictions and checkpoints live under ignored `work/`
or in the verified local archive. Compact verified evidence is published under
[reports](reports/RUN_REPORT.md); full predictions and checkpoints are retained
locally, not committed to Git. Completion is established by the full result
gate and receipts, not merely by passing software tests. The GPU is stopped;
retained cloud storage continues to incur storage charges.

## Rebuild and verify

To verify this recorded execution, decompress the versioned
[frozen plan](evidence/plan.json.gz) and use its accompanying
[capacity](evidence/capacity.json) and
[selection diagnostics](evidence/selection_diagnostics.json), together with the
archived result tree. [PLAN_LOCK.json](evidence/PLAN_LOCK.json) is the unchanged
record at the time of freezing, not a current training-status report. Rebuilding
a plan under a later Git commit produces a different plan identity and is not a
substitute for the plan bound into existing results. The
[result and restoration checks](audit_tools/README.md) describe how to audit the
recorded predictions and checkpoints.

Run from the repository root. Use `python -m v3.speaker_coverage.plan --help` and
`python -m v3.speaker_coverage.verify --help` for complete CLI options. Official
ASV extraction requires an isolated SpeechBrain environment and a pinned local
model snapshot. Training uses the inherited PyTorch environment.
The following planning commands are for constructing and checking a plan; they
do not resume or verify an already completed result archive by themselves.

```sh
python -m v3.speaker_coverage.plan --repo REPO --features FEATURES \
  --asv FEATURES/cremad_ecapa.npz --wavlm-model WAVLM_MODEL --out PLAN_DIR
python -m v3.speaker_coverage.verify --repo REPO --plan PLAN_DIR/plan.json \
  --features FEATURES --asv FEATURES/cremad_ecapa.npz \
  --wavlm-model WAVLM_MODEL --audio-root CREMAD_AUDIO --out GATE_JSON
```

`--draft` produces a planning artifact only. Formal execution requires complete
source/model pins and an independent input gate. `diagnose.py` reads real ASV
features and explicitly identified historical results, never new E2 scores.
The [TTS probe](tts/README.md) has its own manifest and cannot authorize formal
synthetic-arm training without the human quality gate.
