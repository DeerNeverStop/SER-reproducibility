"""Formal GPU entrypoint. Strict lock, exclusive runner, two attempts, deadline, artifact seals."""
import argparse
from datetime import datetime
from pathlib import Path

from . import calib_gpu, common as c
from .execution import LOCK_PATH, run_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default=str(c.DEPLOY_ROOT / "work/plan/B2.json"))
    p.add_argument("--out", default=str(c.DEPLOY_ROOT / "work/B2_gpu"))
    p.add_argument("--lock", type=Path, default=LOCK_PATH)
    p.add_argument("--deadline", required=True)
    p.add_argument("--models", default="cnn,wavlm_ft")
    p.add_argument("--audio-root", default="/workspace/ser-deploy-data/audio")
    p.add_argument("--units", default=None)
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    c.require(a.threads == 8, "formal GPU thread count is fixed at 8")
    a.device, a.audio_profile, a.verify_audio, a.fail_fast = "cuda", "pod", True, True
    deadline = datetime.fromisoformat(a.deadline.replace("Z", "+00:00"))
    c.require(deadline.tzinfo is not None, "deadline requires timezone")
    with run_context(a.out, a.lock, deadline) as guard:
        plan_path = Path(a.plan).resolve()
        c.require(plan_path == (c.DEPLOY_ROOT / "work/plan/B2.json").resolve(), "formal GPU plan must be the locked B2 path")
        rel = plan_path.relative_to(c.REPO_ROOT.resolve()).as_posix()
        c.require(guard.lock["files"].get(rel) == c.sha256_file(plan_path), "GPU plan not bound to execution lock")
        a.run_guard = guard
        result = calib_gpu.cmd_run(a)
        if result == 0 and not a.limit and not a.units and set(a.models.split(",")) == {"cnn", "wavlm_ft"}:
            units = calib_gpu.select_units(c.read_json(plan_path), ["cnn", "wavlm_ft"])
            seals = {}
            for unit in units:
                path = Path(a.out) / "units" / unit["unit_id"] / "SEALED.json"
                c.require(path.is_file(), "missing final managed unit seal")
                seals[unit["unit_id"]] = c.sha256_file(path)
            c.require(len(seals) == 120, "GPU completion requires all 120 units")
            c.atomic_write_json(Path(a.out) / "GPU_COMPLETE.json", {"program": c.PROGRAM, "completed_at": c.now(),
                                "lock_sha256": guard.lock["lock_sha256"], "n_units": 120, "unit_seal_sha256": seals,
                                "scientific_scores_computed": False})
        return result


if __name__ == "__main__":
    raise SystemExit(main())
