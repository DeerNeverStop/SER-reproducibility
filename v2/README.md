# SER v2 one-shot program — how to run

Everything under `v2/` is self-contained (`pip install numpy scipy scikit-learn pandas torch librosa soundfile tqdm pytest`;
`torchaudio` for the SSL caches and the WavLM engines). Paths below are relative to `v2/`.

## 0. Verify the package (any machine, no data)
```bash
python -m pytest tests -q                                   # splits, seeds, stats, registry, plan/score round trip
python -m ser_v2.run_plan --manifests manifests --out plan_rc2 --p1-splits ../results/protocol_premium/splits
python -m ser_v2.fixtures --plan plan_rc2 --manifests manifests --out /tmp/run_syn        # synthetic run (~6 min)
python -m ser_v2.score   --plan plan_rc2 --manifests manifests --run /tmp/run_syn --registry registry
python -m ser_v2.verify  --plan plan_rc2 --manifests manifests --run /tmp/run_syn --registry registry --results /tmp/run_syn/results.json
python -m ser_v2.mutations --plan plan_rc2 --manifests manifests --run /tmp/run_syn --registry registry --work /tmp/mut
```

## 1. PREP (training machine, before tag-1)
1. Build real manifests: `python -m ser_v2.import_p1_manifests --p1-manifests ../results/protocol_premium/manifests --out manifests`
   (RAVDESS, CREMA-D) and `python -c "from ser_v2.corpora import build_manifest; ..."` for the full SUBESCO
   corpus -> `manifests/subesco_manifest.csv` (the 980 panel stays pinned by `registry/subesco_980_pinned.json`).
2. Rebuild the plan (`run_plan` as above); commit `plan_rc2/{run_plan.csv,split_index.json,unit_configs.json,hygiene_log.json,plan_summary.json}`.
3. Caches: `python -m ser_v2.features --corpus cremad --manifest manifests/cremad_manifest.csv --root <audio> --out features --kind logmel`
   and `--kind ssl --encoder hubert_base|wavlm_base_plus|wav2vec2_base` (GPU, ~2.5 h total). Cache names are content-addressed.
4. Dry run on synthetic caches: `python -m ser_v2.features --kind synthetic ...`, then `python -m ser_v2.train --arm CTRL --engine-override synthetic ...`
   and one unit per engine with real engines on synthetic features; run the mutation battery and the scorer/verifier comparison.
5. `git tag -a SER26-prereg-1` (+ OSF registration, OpenTimestamps); record the NIST beacon pulse for the audit order.

## 2. Run (fixed order)
```bash
python -m ser_v2.train --plan plan_rc2 --manifests manifests --features features --run runs/main --arm CTRL
python -m ser_v2.train ... --arm PROBECPU --device cpu          # in parallel on CPU
python -m ser_v2.train ... --arm MECH2X2
python -m ser_v2.train ... --arm HPO
# off-family timing probe (EmoDB / synthetic at CREMA-D cardinality) -> registry cap -> git tag SER26-prereg-2
python -m ser_v2.train ... --arm FT --audio-root <corpus roots>
```
The runner skips DONE units, appends `runs/main/ledger.jsonl`, and stops when an arm cap is reached
(record a deviation before continuing). Conditional units (`conditional=1`) run only if the timing probe allows.

## 3. Score and verify (once)
```bash
python -m ser_v2.score  --plan plan_rc2 --manifests manifests --run runs/main --registry registry
python -m ser_v2.verify --plan plan_rc2 --manifests manifests --run runs/main --registry registry --results runs/main/results.json
```
`results.json` + `numeric_insert.tex` feed the manuscript; `verification.json` must pass with max_abs_diff <= 1e-12.

## 4. Audit
Frame construction uses the GitHub/arXiv APIs on the author's machine (channel A search + capped code-search
channel B), the lineage rule in `ser_v2/audit/protocol.py`, then `ser_v2.audit.sample.priority_order` /
`sequential_screen`, rater forms per `RATER_FORM_SCHEMA`, and `python -m ser_v2.audit.adjudicate --records ... --frame-n N --out ...`.

## What runs where
The rc1 review session had no GPU, no audio and no torchaudio: it produced the plan, split tables, scorer,
verifier, mutation battery, tests and a CPU dry run with a synthetic SUBESCO placeholder. rc2 (this tag) was
prepared on the author's RTX 5070 machine with the real SUBESCO manifest, real log-mel and SSL caches, the
G3 gate re-run on `plan_rc2`, the G4 engine dry runs (including the WavLM engines on synthetic audio), and the
off-family timing probe; see `VALIDATION_RECORD_rc2.md` and `PINS_rc2.json`. Descriptor-only tools that are not
part of the frozen scorer live in `tools/` (`mechid.py` D10/D11, `ridge_sweep.py` D12/D13, `status.py`, drivers).
