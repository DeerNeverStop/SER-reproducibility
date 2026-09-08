"""Partition the pending units of one arm into N disjoint queues of --filter strings, balanced by
estimated cost (times an empirical runtime ratio), grouped by the given plan columns. Each queue is
run sequentially by tools/run_queue.sh; queues never share a unit, so parallel execution is safe.

    python -m tools.make_queues --plan plan_rc2 --run runs/main --arm MECH2X2 --keys corpus_level,model,cell --n 6 --ratio 3.0
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import read_csv  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--keys", default="corpus_level,model")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=3.0)
    a = ap.parse_args(argv)
    keys = a.keys.split(",")
    rows = [r for r in read_csv(a.plan / "run_plan.csv") if r["arm"] == a.arm]
    cost, n = collections.defaultdict(float), collections.Counter()
    for r in rows:
        if (a.run / "units" / r["unit_id"] / "DONE").exists():
            continue
        k = tuple(r[c] for c in keys)
        n[k] += 1
        cost[k] += float(r["est_gpu_sec"]) / 3600 * a.ratio
    Q, load = [[] for _ in range(a.n)], [0.0] * a.n
    for k, h in sorted(cost.items(), key=lambda kv: -kv[1]):
        i = load.index(min(load)); Q[i].append(k); load[i] += h
    a.run.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(Q):
        with open(a.run / f"{a.arm}_queue_{i}.txt", "w", newline="\n") as fh:
            for k in q:
                fh.write(",".join(f"{c}={v}" for c, v in zip(keys, k)) + "\n")
    json.dump({"arm": a.arm, "keys": keys, "queues": [[list(k) for k in q] for q in Q], "projected_hours": load, "pending_units": sum(n.values())},
              open(a.run / f"{a.arm}_queues.json", "w"), indent=1)
    print(json.dumps({"arm": a.arm, "pending_units": sum(n.values()), "projected_hours_per_queue": [round(x, 2) for x in load]}))


if __name__ == "__main__":
    main()
