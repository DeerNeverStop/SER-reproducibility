# N14R independent-verifier mutation protocol

The verifier is an independently implemented replay, not a wrapper around the
scorer.  `verify.py` must never import `n14r.score`, `n14r.plan`, or an N14R/v2
statistics module.  Run the verifier only after the execution runner has written
`completion.json` and the byte-frozen `analysis_lock.json`:

```text
python -m v2_1.n14r.verify --plan <n14r/plan> --run <runs/n14r> \
  --result <results/n14r_results.json> --output <results/n14r_verification.json>
```

A passing report means all preregistration, source, plan, split, ledger,
environment and unit receipts were verified before outcome replay; all 1,920
locked units were reconstructed; and every scorer-result field matched the
independent reconstruction.  Any mutation below must produce a non-zero exit.

## Required mutations

| ID | Mutation | Re-seal downstream hashes? | Required rejection |
|---|---|---:|---|
| M01 | Remove one config (for example index 6) from an HPO episode | No | Episode is not exactly config indices 0--7 |
| M02 | Change one byte in `predictions.csv` | No | Analysis-lock artifact hash mismatch |
| M03 | Delete one outer-test prediction row | Yes: update prediction SHA in `unit.json` and `DONE`, then update the test lock copy | Predictions are not an exact frozen test cover |
| M04 | Change `train_seed` in `unit.json` | Yes: update the locked `unit.json` hash | Unit/plan train-seed mismatch |
| M05 | Change a split path, label, speaker, or sample index | Rebuild superficial file/index hashes only | Manifest join, speaker separation, six-class support, outer pairing, or scientific partition hash mismatch |
| M06 | Change a plan config payload/hash or `unit_configs.json` | Rebuild superficial file hashes only | Frozen grid/config/unit-ID commitment mismatch |
| M07 | Replace `history.json` or alter its first minimum-loss epoch/UAR | Yes: update the lock artifact hash | `best_epoch`, `val_loss_best`, or `val_uar_best` history mismatch |
| M08 | Change one draw value in scorer JSON | Yes, if a result digest exists outside this protocol | Recursive mismatch at `$.primary.values[i]` and/or its draw field |
| M09 | Change `primary.p_two_sided` or `primary.p_holm` | Yes | Recursive p-value mismatch |
| M10 | Change `primary.verdict`/`supported` | Yes | Recursive decision mismatch |
| M11 | Add an unrecognized scorer-result field | N/A | Extra field mismatch; verifier ignores no scorer field |
| M12 | Change `spec.json`, `PINS.json`, a pinned source, plan file, or external feature cache | No | Preregistration/source/environment hash gate |
| M13 | Remove a locked unit, alter `DONE`, or substitute an artifact from another unit | No | Exact 1,920-unit set or four-file receipt mismatch |
| M14 | Reorder/skip reserve draws, disagree on void draws, or put an outcome statistic before `analysis_lock` | Re-hash ledger/lock if testing semantic protection | Frozen reserve order, completion/ledger agreement, or no-interim-outcomes rule |

“Re-seal” mutations are essential: they demonstrate that the verifier checks the
scientific meaning behind a hash chain instead of merely noticing stale hashes.
The checked-in pytest mutations implement M01--M04 and M08--M11 directly.  M05--
M07 and M12--M14 are release-candidate acceptance tests because they operate on
the complete 2,240-row plan/1,920-unit locked bundle.

## Release-candidate procedure

1. Copy the complete bundle to a temporary directory; never mutate the evidence
   source in place.
2. Run the scorer and verifier once without mutation.  Preserve both JSON files
   and their SHA-256 values as the baseline.
3. Apply exactly one mutation per fresh copy.  Where the table says “re-seal”,
   update only the named superficial receipts so the intended semantic check is
   reached; do not regenerate the scientific plan or expected result.
4. Require a non-zero verifier exit and retain the first reported mismatch path
   or integrity error.  A crash, timeout, warning-only response, or “not tested”
   downgrade is not an acceptable rejection.
5. Restore from the clean copy and confirm its byte hashes and passing report are
   unchanged.  Do not publish a release unless all fourteen mutations reject.

The full exact sign-flip replay enumerates all 16,777,216 assignments.  Unit tests
may disable that expensive step only when exercising a pre-inference integrity
gate; release verification must use the default full enumeration and 100,000
bootstrap replicates.
