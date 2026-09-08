"""Stage snapshot for the per-stage commits Codex asked for: counts, gpu-hours, ledger and
deviation hashes, and the SHA-256 of every DONE unit's predictions.csv for one arm.

    python -m tools.stage_snapshot --plan plan_rc2 --run runs/main --arm CTRL --out evidence/main/stage_CTRL.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import atomic_write_json, read_csv, read_json, sha256_file  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = [r for r in read_csv(a.plan / "run_plan.csv") if r["arm"] == a.arm]
    per_level = defaultdict(lambda: {"units": 0, "done": 0, "gpu_hours": 0.0})
    shas = {}
    gpu = 0.0
    for r in rows:
        g = per_level[r["corpus_level"]]
        g["units"] += 1
        udir = a.run / "units" / r["unit_id"]
        if (udir / "DONE").exists():
            u = read_json(udir / "unit.json")
            g["done"] += 1
            g["gpu_hours"] += float(u.get("gpu_seconds", 0.0)) / 3600
            gpu += float(u.get("gpu_seconds", 0.0)) / 3600
            shas[r["unit_id"]] = u["predictions_sha256"]
    dev = (a.run / "deviations.jsonl")
    snap = {"arm": a.arm, "at": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha256_file(a.plan / "run_plan.csv"),
            "units": len(rows), "done": len(shas), "gpu_hours_measured": round(gpu, 3),
            "per_level": {k: {**v, "gpu_hours": round(v["gpu_hours"], 3)} for k, v in sorted(per_level.items())},
            "ledger_sha256": sha256_file(a.run / "ledger.jsonl") if (a.run / "ledger.jsonl").exists() else None,
            "deviations_sha256": sha256_file(dev) if dev.exists() else None,
            "deviations": [json.loads(l) for l in dev.read_text(encoding="utf-8").splitlines() if l.strip()] if dev.exists() else [],
            "predictions_sha256": shas}
    atomic_write_json(a.out, snap)
    print(json.dumps({k: v for k, v in snap.items() if k not in ("predictions_sha256", "per_level", "deviations")}, indent=1))


if __name__ == "__main__":
    main()
