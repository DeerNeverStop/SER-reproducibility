# Independent result and resource checks

These tools were written and tested before unsealing the new E2 scientific
results. They do not alter the frozen execution source or select a new endpoint.
The experiment remains bound to plan `8ea69b57...` and source commit `db7836c...`.

The [completed run report](../reports/RUN_REPORT.md) records the actual checks:
720 formal and 19 excluded pilot units passed, and the independent numerical
replay matched all 12 contrasts and five CSV tables (25,703 numeric values,
maximum absolute error below 2.85e-14). The resource audit found no failed or
unclosed fit attempts. The figure title was clarified after scoring to specify
576 SER training recordings per fit; this cosmetic change did not alter the
frozen analysis or inputs.

`replay_coverage_results.py` requires all 720 formal units, checks the frozen
integrity verifier's checkout and source identity, then independently reconstructs
the six-class, six-rotation person-level aggregation from the saved logits. It
uses bootstrap count weights to cross-check the declared 10,000 paired-person
resamples, all 12 contrasts, policy/person/repeat/fold tables and primary result.
It does not import the main scoring implementation. Its 20 synthetic tests cover
the complete factorial grid, varying query supports, ties, and output tampering.

`resource_coverage_results.py` requires accepted complete pilot/formal integrity
receipts, and binds them to current DONE markers, receipts, histories and ledgers.
It summarizes all 739 successful runs and any failed/incomplete attempts without
opening prediction arrays. Its 14 synthetic tests check this accounting and
tampering. Large payload integrity comes from the accepted seals; the resource
tool does not repeat tensor replay. Per-fit times are not GPU rental duration or
an invoice. Those quantities need separate provider records.

`plot_coverage_results.py` renders the prespecified comparisons and policy means
from the complete score report; it adds no statistical inference. The optional
synthetic preview is prominently labelled and is not experimental evidence.

`restore_coverage_checkpoints.py` fixes nine formal sentinel units before score
unsealing: fold 0, rotation 0, draw 0, all three policies and models (FT seed 0).
It restores both stored states in a separate process and compares their logits,
without fitting or computing accuracy. It shares the frozen architectures and
inference functions, so it is a process/restoration check, not an independent
model implementation. Eight synthetic CPU checks passed. The real check must use
the original training virtual environment, package versions, and RTX 5090. Its
resolved interpreter identity may be `/usr/bin/python3.12`; invoke the original
virtual-environment entry point to retain its installed packages. Neural inference
is FP32, matching the runner; float16 AMP was used during FT training only.

The [recorded real restoration check](../reports/data/gates/restore9.json) passed
on 2026-09-06: all nine fixed units reproduced both saved states' original logits
exactly (maximum absolute difference 0). Total wall time was 12.96 seconds, with
2.00 seconds of measured inference. This is a technical restoration result, not
a SER score or an independent replication of the scientific experiment. It was
completed before the formal performance results were unsealed.

From the repository root, use a Python environment with the experiment's required
NumPy/PyTorch dependencies (Matplotlib is additionally needed for figures):

```bash
python -m v3.speaker_coverage.verify --repo . --plan /path/plan.json \
  --outroot /path/runs --phase formal --block all --consolidate --out /path/formal_gate.json
python -m v3.speaker_coverage.score --repo . --plan /path/plan.json \
  --outroot /path/runs --out /path/scores
python v3/speaker_coverage/audit_tools/replay_coverage_results.py \
  --repo . --plan /path/plan.json --outroot /path/runs \
  --results /path/scores --out /outside-inputs/numeric_replay.json
python v3/speaker_coverage/audit_tools/resource_coverage_results.py \
  --repo . --plan /path/plan.json --outroot /path/runs \
  --pilot-gate /path/pilot_gate.json --formal-gate /path/formal_gate.json \
  --out /outside-repository-and-runs/resource_audit.json
python v3/speaker_coverage/audit_tools/plot_coverage_results.py \
  --results /path/scores/results.json --out /path/figures
```

Run the separate restoration check while the first FT shard is still on the
original pod, before its verified offload/cleanup. Always provide `--repo`:

```bash
/workspace/ser-deploy-env/bin/python /workspace/ops-coverage/restore_coverage_checkpoints.py \
  --repo /workspace/ser-coverage --plan /workspace/coverage-plan/plan.json \
  --results-root /workspace/coverage-runs --features /workspace/coverage-features \
  --audio-root /workspace/ser-deploy-data/audio/cremad \
  --wavlm-model /workspace/torch-cache/hub/checkpoints/wavlm_base_plus.pth \
  --out /workspace/ops-coverage/restore9 --model all --threads 1
```

Keep the integrity gates and audit outputs outside the inputs they verify. The
raw checkpoint/prediction archive remains local; small verified results and
receipts are copied into `reports/` for review. These checks address execution and
numerical consistency, not the truth of a mechanism or generalization to another
corpus/language.
