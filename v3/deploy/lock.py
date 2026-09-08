'''Plan lock for SER26-DEPLOY-1: pins plans, code, split index, manifests, feature caches and environment.

python -m v3.deploy.lock write  --out v3/deploy/PLAN_LOCK.json
python -m v3.deploy.lock verify --lock v3/deploy/PLAN_LOCK.json
'''
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from . import common as c

CODE_FILES = ["v3/deploy/SPEC_DEPLOY_ZH.md", "v3/deploy/PRIOR_EXPOSURE.md", "v3/deploy/common.py", "v3/deploy/enroll.py",
              "v3/deploy/review_budget.py", "v3/deploy/calib.py", "v3/deploy/calib_gpu.py", "v3/deploy/engines_deploy.py",
              "v3/deploy/verify.py", "v3/deploy/lock.py", "v2/ser_v2/train.py", "v2/ser_v2/features.py", "v2/ser_v2/stats.py",
              "v2/ser_v2/corpora.py", "advanced_models.py", "v2/plan_rc2/split_index.json", "v2/plan_rc2/unit_configs.json"]
PLAN_FILES = ["v3/deploy/work/plan/A.json", "v3/deploy/work/plan/B2.json"]
MANIFESTS = ["v2/manifests/ravdess_manifest.csv", "v2/manifests/cremad_manifest.csv", "v2/manifests/subesco_manifest.csv"]
SPLIT_KEYS = [f"ctrl__{lvl}__{cell}__r{r}" for lvl in c.LEVELS for cell in ("GG", "GR") for r in range(3)]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=c.REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    import numpy, sklearn
    env = {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__, "scikit-learn": sklearn.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        env["torch"] = None
    return env


def build() -> dict:
    files = {}
    for rel in CODE_FILES + PLAN_FILES + MANIFESTS:
        p = c.REPO_ROOT / rel
        files[rel] = c.sha256_file(p) if p.is_file() else None
    idx = c.split_index()
    splits = {k: {"index_sha256": idx[k]["sha256"], "local_sha256": c.sha256_file(c.V2_DATA_ROOT / "plan_rc2" / idx[k]["path"])} for k in SPLIT_KEYS}
    caches = {}
    for base in ("ravdess", "cremad", "subesco"):
        for kind in list(c.ENCODERS) + ["logmel"]:
            cands = sorted((c.V2_DATA_ROOT / "features").glob(f"{base}__{kind}__*.npz"))
            caches[f"{base}/{kind}"] = {"file": cands[0].name, "sha256": c.sha256_file(cands[0])} if len(cands) == 1 else {"error": f"{len(cands)} candidates"}
    return {"schema": "ser26-deploy-plan-lock-1", "program": c.PROGRAM, "created_at": c.now(), "git_head": git_head(),
            "scientific_outcomes_seen": False, "files": files, "splits": splits, "feature_caches": caches, "environment": environment()}


def main(argv=None):
    raise SystemExit("DEPLOY-1 lock CLI is historical and disabled. Use python -m v3.deploy.execution write/verify for DEPLOY-2.")


if __name__ == "__main__":
    sys.exit(main())
