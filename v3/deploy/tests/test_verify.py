"""Tests for the independent verifier: positive check against a naive second implementation, plus a mutation battery."""
from __future__ import annotations

import csv
import gzip
import json
import shutil
from pathlib import Path

import pytest

from v3.deploy import verify as V
from v3.deploy.tests import fixture_deploy as fx, naive_scorer as ns

MODULES = ["A", "B1", "B2"]


def score_with_naive(f):
    f["level_map"] = dict(fx.LEVELS)
    plan_a = json.load(open(f["plan_dir"] / "A.json", encoding="utf-8"))
    plan_b2 = json.load(open(f["plan_dir"] / "B2.json", encoding="utf-8"))
    ep, ab, ps = ns.score_a(f, plan_a, fx.CONDITIONS, fx.TEST_SPK, fx.ENC)
    ns.write_results(f["results_dir"], "A", ep, ab, ps, V.A_PS_COLS)
    rows = [r for r in csv.DictReader(open(f["run_plan"], encoding="utf-8")) if r["cell"] == "GG"]
    ep, ps = ns.score_b1(f, [(r["corpus_level"], r["model"], r["unit_id"]) for r in rows])
    ns.write_results(f["results_dir"], "B1", ep, [], ps, V.B1_PS_COLS)
    ep, ab, ps = ns.score_b2(f, plan_b2)
    ns.write_results(f["results_dir"], "B2", ep, ab, ps, V.B2_PS_COLS)


def run_verify(f, modules=MODULES, tolerance=1e-9):
    src = V.Sources(f["manifests_dir"], f["split_index"], f["v2root"], f["run_plan"], dict(fx.LEVELS))
    return V.run(modules, f["plan_dir"], f["run_dir"], f["results_dir"], f["root"] / "verification.json", src, tolerance)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("deploy_fixture")
    f = fx.build_study(root)
    score_with_naive(f)
    return f


@pytest.fixture
def clone(study, tmp_path):
    """Fresh copy of the passing study for a mutation."""
    dst = tmp_path / "study"
    shutil.copytree(study["root"], dst)
    f = dict(study)
    for k in ("root", "v2root", "plan_dir", "run_dir", "results_dir", "manifests_dir", "split_index", "run_plan"):
        f[k] = dst / Path(study[k]).relative_to(study["root"])
    return f


def test_positive_pass(study):
    rep = run_verify(study)
    assert rep["pass"], rep["reasons"]
    for m in MODULES:
        blk = rep["modules"][m]
        assert blk["integrity"]["pass"] and blk["comparison"]["pass"]
        assert blk["comparison"]["compared"] > 0 and blk["comparison"]["one_side_only"] == 0
        assert (study["results_dir"] / f"verification_recomputed_{m}_endpoints.csv").is_file()
        assert (study["results_dir"] / f"verification_recomputed_{m}_per_speaker.csv").is_file()
    out = json.loads((study["root"] / "verification.json").read_text(encoding="utf-8"))
    assert out["pass"] is True and out["ambiguities"]


def failing(f, module, check_substr):
    rep = run_verify(f, [module])
    assert rep["pass"] is False
    reasons = " ".join(rep["reasons"])
    assert check_substr in reasons, reasons
    return rep


def a_unit_dir(f):
    return f["run_dir"] / "A" / "units" / f["a_units"][0]["unit_id"]


def rewrite_gz(udir, transform, fix_done=True):
    with gzip.open(udir / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        lines = fh.read().splitlines()
    lines = transform(lines)
    with gzip.open(udir / "predictions.csv.gz", "wt", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    if fix_done:
        (udir / "DONE").write_text(fx.sha_file(udir / "predictions.csv.gz") + "\n", encoding="utf-8")


def test_flipped_label(clone):
    def flip(lines):
        parts = lines[1].split(","); parts[5] = str(1 - int(parts[5])); lines[1] = ",".join(parts); return lines
    rewrite_gz(a_unit_dir(clone), flip)
    failing(clone, "A", "label_mismatch")


def test_dropped_row(clone):
    rewrite_gz(a_unit_dir(clone), lambda lines: lines[:1] + lines[2:])
    failing(clone, "A", "row_set")


def test_duplicated_row(clone):
    rewrite_gz(a_unit_dir(clone), lambda lines: lines + [lines[1]])
    failing(clone, "A", "duplicate_rows")


def test_wrong_done_sha(clone):
    (a_unit_dir(clone) / "DONE").write_text("0" * 64 + "\n", encoding="utf-8")
    failing(clone, "A", "DONE_sha")


def test_unit_missing(clone):
    shutil.rmtree(a_unit_dir(clone))
    failing(clone, "A", "missing_file")


def test_b2_val_test_swapped(clone):
    udir = clone["run_dir"] / "B2" / "units" / clone["b2_units"][0]["unit_id"]
    v, t = (udir / "val_predictions.csv").read_text(encoding="utf-8"), (udir / "test_predictions.csv").read_text(encoding="utf-8")
    (udir / "val_predictions.csv").write_text(t, encoding="utf-8", newline="\n"); (udir / "test_predictions.csv").write_text(v, encoding="utf-8", newline="\n")
    (udir / "DONE").write_text(f"{fx.sha_file(udir / 'val_predictions.csv')} {fx.sha_file(udir / 'test_predictions.csv')}\n", encoding="utf-8")
    failing(clone, "B2", "row_set")


def test_b1_predictions_sha(clone):
    rows = [r for r in csv.DictReader(open(clone["run_plan"], encoding="utf-8")) if r["cell"] == "GG"]
    p = clone["v2root"] / "runs" / "main" / "units" / rows[0]["unit_id"] / "predictions.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
    failing(clone, "B1", "predictions_sha256")


@pytest.mark.parametrize("module", MODULES)
def test_endpoint_perturbed(clone, module):
    p = clone["results_dir"] / module / "endpoints.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["endpoints"][0]["estimate"] += 1e-6
    p.write_text(json.dumps(d), encoding="utf-8")
    rep = failing(clone, module, "comparison")
    assert rep["modules"][module]["comparison"]["failed"] == 1


@pytest.mark.parametrize("module", MODULES)
def test_ci_perturbed(clone, module):
    p = clone["results_dir"] / module / "endpoints.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["endpoints"][0]["ci95"][1] += 1e-6
    p.write_text(json.dumps(d), encoding="utf-8")
    failing(clone, module, "comparison")


@pytest.mark.parametrize("module", MODULES)
def test_per_speaker_row_removed(clone, module):
    p = clone["results_dir"] / module / "per_speaker.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8", newline="\n")
    rep = failing(clone, module, "comparison")
    assert rep["modules"][module]["comparison"]["one_side_only"] >= 1
