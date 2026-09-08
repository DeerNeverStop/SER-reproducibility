"""Progress summary of a run directory against the plan: DONE / failed / pending per arm and
level, GPU-seconds spent, and the last ledger events.

    python -m tools.status --plan plan_rc2 --run runs/main
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import read_csv, read_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--by-level", action="store_true")
    a = ap.parse_args(argv)
    plan = read_csv(a.plan / "run_plan.csv")
    failed = set()
    ledger = a.run / "ledger.jsonl"
    events = []
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                events.append(e)
                if e.get("event") == "failed":
                    failed.add(e["unit_id"])
                elif e.get("event") == "done":
                    failed.discard(e["unit_id"])
    agg = defaultdict(lambda: {"units": 0, "done": 0, "failed": 0, "est_h": 0.0, "spent_h": 0.0})
    for r in plan:
        key = (r["arm"], r["corpus_level"]) if a.by_level else (r["arm"],)
        g = agg[key]
        g["units"] += 1
        g["est_h"] += float(r["est_gpu_sec"]) / 3600
        udir = a.run / "units" / r["unit_id"]
        if (udir / "DONE").exists():
            g["done"] += 1
            try:
                g["spent_h"] += float(read_json(udir / "unit.json").get("gpu_seconds", 0.0)) / 3600
            except Exception:  # noqa: BLE001
                pass
        elif r["unit_id"] in failed:
            g["failed"] += 1
    for key in sorted(agg):
        g = agg[key]
        print(f"{' | '.join(key):32s} done {g['done']:5d}/{g['units']:5d}  failed {g['failed']:3d}  est {g['est_h']:6.2f} h  spent {g['spent_h']:6.2f} h")
    print("last events:")
    for e in events[-5:]:
        print("  ", json.dumps(e)[:160])


if __name__ == "__main__":
    main()
