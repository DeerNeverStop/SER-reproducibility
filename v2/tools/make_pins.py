"""Write PINS_<tag>.json: SHA-256 of every file frozen at tag-1 (registry, plan, spec, scorer,
verifier, split/seed/plan generators, statistics, runner, features) plus the manifests.

    python -m tools.make_pins --plan plan_rc2 --out PINS_rc2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import sha256_file  # noqa: E402

FROZEN = ["registry/hypothesis_registry.csv", "registry/claim_map.json", "registry/arms.json", "registry/power_table.json",
          "registry/subesco_980_pinned.json", "SPEC_SCORING_CONTRACT.md", "PREREGISTRATION_V2.md",
          "ser_v2/score.py", "ser_v2/verify.py", "ser_v2/splits.py", "ser_v2/seeds.py", "ser_v2/run_plan.py", "ser_v2/stats.py",
          "ser_v2/train.py", "ser_v2/features.py", "ser_v2/corpora.py", "ser_v2/registry.py", "ser_v2/common.py",
          "manifests/ravdess_manifest.csv", "manifests/cremad_manifest.csv", "manifests/subesco_manifest.csv", "manifests/subesco_980_manifest.csv"]
PLAN = ["run_plan.csv", "split_index.json", "unit_configs.json", "hygiene_log.json", "plan_summary.json"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    pins = {}
    for rel in FROZEN + [f"{a.plan}/{p}" for p in PLAN]:
        pins[rel] = sha256_file(V2 / rel)
    a.out.write_text(json.dumps(pins, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(pins, indent=1))


if __name__ == "__main__":
    main()
