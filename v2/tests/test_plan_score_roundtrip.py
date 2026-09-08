"""End-to-end on a tiny synthetic program: plan -> fixture run -> score -> mutation."""
import json
from pathlib import Path

import numpy as np

from ser_v2 import corpora, fixtures, run_plan
from ser_v2.common import read_csv, write_csv
from ser_v2.score import Run, Scorer

ROOT = Path(__file__).resolve().parents[1]


def test_plan_is_deterministic_and_scores(tmp_path: Path):
    man = tmp_path / "manifests"; man.mkdir()
    for corpus, spec in (("ravdess", (24, 2, 4)), ("cremad", (91, 12, 1)), ("subesco", (20, 10, 5))):
        rows = corpora.synthetic_manifest(corpus, *spec)
        write_csv(man / f"{corpus}_manifest.csv", corpora.MANIFEST_FIELDS, rows)
    s1 = run_plan.build(man, tmp_path / "plan_a", synthetic=False)
    s2 = run_plan.build(man, tmp_path / "plan_b", synthetic=False)
    assert s1["run_plan_sha256"] == s2["run_plan_sha256"]
    assert s1["within_program_cap"]
    rows = read_csv(tmp_path / "plan_a" / "run_plan.csv")
    assert len({r["unit_id"] for r in rows}) == len(rows)
    # tiny fixture: ravdess CTRL only, then score
    fixtures.generate(tmp_path / "plan_a", man, tmp_path / "run", fixtures.load_world(None), {"CTRL"}, {"ravdess"})
    res = Scorer(Run(tmp_path / "plan_a", man, tmp_path / "run", ROOT / "registry")).run_all()
    assert all(v["pass"] for v in res["integrity"].values())
    assert res["hypotheses"]["R01"]["tested"] and res["hypotheses"]["R01"]["mean"] > 0
    assert res["hypotheses"]["N01"]["verdict"] == "not tested" and res["hypotheses"]["N01"]["m"] == 4
    assert res["hypotheses"]["N11"]["a_priori_status"] == "estimate" and res["hypotheses"]["N11"]["p_holm"] is None
    assert res["claims"]["C1"]["verdict"].startswith("reported")
