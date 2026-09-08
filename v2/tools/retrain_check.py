"""STAT-stage bitwise retrain check (preregistration section 7: "rolling 1 %+1 % bitwise retrain checks").

Selects 1 % of the GPU units (CTRL, MECH2X2, HPO, FT) and 1 % of the CPU units (PROBECPU) by a
deterministic hash order, re-runs each with the frozen runner into a separate run directory, and
compares the SHA-256 of predictions.csv with the main run. Any mismatch is reported verbatim.

    python -m tools.retrain_check --plan plan_rc2 --run runs/main --check runs/retrain_check --select      # writes the selection
    python -m tools.retrain_check --plan plan_rc2 --run runs/main --check runs/retrain_check --compare     # after the re-runs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import atomic_write_json, read_csv, read_json  # noqa: E402

GPU_ARMS = ("CTRL", "MECH2X2", "HPO", "FT")
AUDIO_ROOT = {"ravdess": "E:/科研/SER/data", "cremad": "E:/科研/SER/AudioWAV", "subesco_980": "E:/claudework_data/ICASSP2027-corpora/subesco/extracted"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--check", type=Path, required=True)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--fraction", type=float, default=0.01)
    a = ap.parse_args(argv)
    plan = read_csv(a.plan / "run_plan.csv")
    a.check.mkdir(parents=True, exist_ok=True)
    sel_path = a.check / "selection.json"
    if a.select:
        key = lambda r: hashlib.sha256(("retrain|" + r["unit_id"]).encode()).hexdigest()
        gpu = sorted([r for r in plan if r["arm"] in GPU_ARMS], key=key)
        cpu = sorted([r for r in plan if r["arm"] == "PROBECPU"], key=key)
        n_gpu, n_cpu = math.ceil(a.fraction * len(gpu)), math.ceil(a.fraction * len(cpu))
        chosen = gpu[:n_gpu] + cpu[:n_cpu]
        sel = [{"unit_id": r["unit_id"], "arm": r["arm"], "corpus_level": r["corpus_level"], "model": r["model"], "cell": r["cell"], "fold": r["fold"],
                "device": "cpu" if r["arm"] == "PROBECPU" else "cuda",
                "audio_root": AUDIO_ROOT.get(r["corpus_level"]) if r["arm"] == "FT" else None} for r in chosen]
        atomic_write_json(sel_path, {"fraction": a.fraction, "n_gpu_pool": len(gpu), "n_cpu_pool": len(cpu), "n_selected": len(sel), "units": sel})
        with open(a.check / "commands.sh", "w", newline="\n", encoding="utf-8") as fh:
            fh.write("#!/usr/bin/env bash\nset -u\ncd \"$(dirname \"$0\")/../..\"\nP=\"E:/科研/SER/ser_gpu/Scripts/python.exe\"\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n")
            for u in sel:
                extra = f' --audio-root "{u["audio_root"]}"' if u["audio_root"] else ""
                fh.write(f'"$P" -m ser_v2.train --plan {a.plan.as_posix()} --manifests manifests --features features --run {a.check.as_posix()} --arm {u["arm"]} --device {u["device"]} --filter unit_id={u["unit_id"]}{extra} >> {a.check.as_posix()}/train.log 2>&1\n')
            fh.write('echo "RETRAIN DONE $(date)" >> ' + a.check.as_posix() + '/train.log\n')
        print(json.dumps({"n_selected": len(sel), "by_arm": {arm: sum(1 for u in sel if u["arm"] == arm) for arm in GPU_ARMS + ("PROBECPU",)}}))
    if a.compare:
        sel = read_json(sel_path)["units"]
        rows, n_same, n_diff, n_missing = [], 0, 0, 0
        for u in sel:
            main_u = a.run / "units" / u["unit_id"] / "unit.json"
            chk_u = a.check / "units" / u["unit_id"] / "unit.json"
            if not chk_u.exists():
                n_missing += 1; rows.append({**u, "status": "missing"}); continue
            m, c = read_json(main_u), read_json(chk_u)
            same = m["predictions_sha256"] == c["predictions_sha256"]
            n_same += same; n_diff += (not same)
            row = {**u, "status": "identical" if same else "DIFFERENT", "main_sha": m["predictions_sha256"], "check_sha": c["predictions_sha256"],
                   "main_best_epoch": m.get("best_epoch"), "check_best_epoch": c.get("best_epoch"), "main_val_uar": m.get("val_uar_best"), "check_val_uar": c.get("val_uar_best")}
            if not same:
                import csv
                import numpy as np
                from ser_v2 import stats
                def load(p):
                    rr = list(csv.reader(open(p, encoding="utf-8")))[1:]
                    return np.array([int(x[3]) for x in rr]), np.array([int(x[4]) for x in rr]), np.array([[float(v) for v in x[5:]] for x in rr])
                y, p1, l1 = load(a.run / "units" / u["unit_id"] / "predictions.csv"); _, p2, l2 = load(chk_u.parent / "predictions.csv")
                K = l1.shape[1]
                row.update({"n_test": int(len(y)), "y_pred_agreement": float((p1 == p2).mean()), "uar_main": stats.uar(y, p1, K), "uar_check": stats.uar(y, p2, K),
                            "max_abs_logit_diff": float(np.abs(l1 - l2).max()), "engine": m.get("config", {}).get("engine"), "fp16": m.get("config", {}).get("fp16")})
            rows.append(row)
        rep = {"n_selected": len(sel), "identical": n_same, "different": n_diff, "missing": n_missing, "bitwise_pass": n_diff == 0 and n_missing == 0, "units": rows}
        atomic_write_json(a.check / "retrain_report.json", rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "units"}))
        for r in rows:
            if r["status"] != "identical":
                print(json.dumps(r))


if __name__ == "__main__":
    main()
