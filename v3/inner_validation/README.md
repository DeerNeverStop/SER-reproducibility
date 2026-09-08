# Common-fit validation-speaker exposure

Completed on 2026-09-07: **480 formal fits + four separate technical pilots**.
See the [final report](reports/RUN_REPORT.md), [exact results](reports/results/results.json),
and [completed evidence](evidence/completed/FILE_SHA256.json).
The primary relative validation-optimism difference is **+2.706 pp**
(95% t CI **[1.802, 3.610]**, 24 draws).
The selected seen-rule model has a descriptively higher test mean (+0.847 pp);
this study does not establish that unseen-speaker selection improves test accuracy.
Both complete artifact gates and independent numerical replay passed; the GPU
was confirmed stopped before scientific scoring.

This is a separate experiment after the completed
[speaker-coverage study](../speaker_coverage/reports/RUN_REPORT.md).
The [protocol](PROTOCOL.md) tests validation optimism on a modern WavLM model
while holding its actual fitting trajectory and external test fixed.

There are 24 draws × five folds × four learning-rate configurations = 480 formal
fits, plus four isolated technical pilot fits. Each fit supplies a seen-speaker
and an unseen-speaker validation selection. Their performance is compared on the
same held-out speakers and texts. This is a controlled extension, not a relabelled
N14R2 random/grouped partition replication or a new test of the old coverage arms.

The scientific source was frozen at `de39f2ae5bad4540da9d13ab4cc6196bec786400`;
46 scientific synthetic tests passed on the cloud before the technical pilot.
Later audit and operations helpers have separate tests and source records.
The independent numerical replay checked 35,744 values with maximum absolute
error 1.421e-14. These software/artifact checks do not certify experimental
generality or a causal speaker mechanism. Private Git freezing is not public
preregistration.

Use the archived frozen plan to verify an execution; generating a new plan under
a later source commit changes its identity. Work products are kept under ignored
`work/` and the separate local `D:/SER-dual-validation-20260906` archive.
Raw audio, predictions and large checkpoints are not Git artifacts.

```bash
# /path/plan.json must contain the original bytes from evidence/plan.json.gz.
python -m v3.inner_validation.run input-gate --repo . --plan /path/plan.json \
  --audio-root /path/AudioWAV --model /path/wavlm_base_plus.pth --out /path/input_gate.json
python -m v3.inner_validation.run run --repo . --plan /path/plan.json \
  --audio-root /path/AudioWAV --model /path/wavlm_base_plus.pth \
  --outroot /path/runs --phase pilot
python -m v3.inner_validation.run result-gate --repo . --plan /path/plan.json \
  --outroot /path/runs --phase formal --out /path/formal_gate.json
python -m v3.inner_validation.score --repo . --plan /path/plan.json \
  --outroot /path/runs --gate /path/formal_gate.json --out /path/scores
python -m v3.inner_validation.audit_tools.replay --repo . --plan /path/plan.json \
  --outroot /path/runs --gate /path/formal_gate.json --results /path/scores \
  --out /path/replay.json
```

Formal batches use the same `run run` command with `--phase formal` and a frozen
`--unit-ids` list when storage requires offloading. Only the full 480-unit local
gate permits scientific scoring. Checkpoints, losses and both validation rules
remain fixed regardless of the direction of results.
