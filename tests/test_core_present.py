"""Generated TEST data only; never open real Study II analysis or predictions."""
import csv
import hashlib
from pathlib import Path

import pytest

from v3.data_design import core_present as present


def TEST_report():
    def estimate(value):
        return {"estimate_pp": value, "ci95_percentile_pp": [value - 2.4, value + 3.1], "n_speakers": 91,
                "draw_estimates_pp": [value - 1., value + .25, value + .75],
                "draw_range_pp": [value - 1., value + .75]}
    absolute = []
    for i, (model, budget, count, scenario) in enumerate(present.POLICIES):
        absolute.append({"model": model, "B": budget, "S": count, "scenario": scenario, **estimate(37. + i)})
    contrasts = []
    for i, (model, budget, contrast, scenario) in enumerate(present.CONTRASTS):
        row = {"model": model, "B": budget, "contrast": contrast,
               "primary": scenario == "prompt_new", **estimate(float((i % 5) - 2))}
        if scenario:
            row["scenario"] = scenario
        contrasts.append(row)
    return {"schema": "ser-study2-score-1", "absolute_uar": absolute, "contrasts": contrasts}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(present.canonical(value) + b"\n")


@pytest.fixture
def approved_TEST(tmp_path, monkeypatch):
    repo, run, analysis = tmp_path / "repo", tmp_path / "run", tmp_path / "analysis"
    plan_path = repo / "plan.json"
    source_dir = repo / "v3/data_design"
    for name in ("core_score.py", "core_verify.py", "result_verify.py", "core_present.py"):
        path = source_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((Path(present.__file__).parent / name).read_bytes())
    for name in ("manifest.csv", "demographics.csv"):
        (repo / name).write_text("TEST metadata\n", encoding="utf-8")
    plan = {"input": {"manifest_path": "manifest.csv", "demographics_path": "demographics.csv"},
            "source_sha256": {"v3/data_design/core_verify.py": present.sha(source_dir / "core_verify.py")},
            "design": {"runtime_complete": True}, "units": []}
    plan["plan_sha256"] = hashlib.sha256(present.canonical(plan)).hexdigest()
    write_json(plan_path, plan)
    for name in ("completion.json", "analysis_lock.json", "run_identity.json", "ledger.jsonl",
                 "environment_ridge_wavlm.json", "environment_cnn.json"):
        write_json(run / name, {"TEST": name})
    closure = {"plan_sha256": plan["plan_sha256"], "verified_units": 1440,
               "analysis_lock_sha256": present.sha(run / "analysis_lock.json"),
               "completion_sha256": present.sha(run / "completion.json"),
               "ledger_sha256": present.sha(run / "ledger.jsonl"), "done_mapping_sha256": "a" * 64}
    identity = {**closure, "scorer_sha256": present.sha(source_dir / "core_score.py"),
                "numpy_version": "TEST", "python_version": "TEST"}
    report = TEST_report()
    report["identity"] = identity
    analysis.mkdir()
    (analysis / "per_speaker.csv").write_bytes(b"TEST opaque CSV; presentation does not recompute it\n")
    report["per_speaker_sha256"] = present.sha(analysis / "per_speaker.csv")
    write_json(analysis / "score_identity.json", identity)
    write_json(analysis / "score.json", report)
    write_json(analysis / "SCORE_DONE", {"identity": identity, "score_sha256": present.sha(analysis / "score.json"),
                                     "per_speaker_sha256": report["per_speaker_sha256"]})
    paths = present.required_snapshot_paths(repo, plan_path, run, analysis, plan)
    approval = {"schema": "ser-study2-independent-result-verification-1", "pass": True,
                "results_approved_for_paper": True, "scientific_arrays_read": True, "numeric_fields_checked": 100,
                "tolerance_pp": 1e-10, "max_numeric_error": 0., "identity": identity,
                "n_units": 1440, "n_speakers": 91, "absolute_estimates": 16, "speaker_count_contrasts": 8,
                "interaction_contrasts": 4, "per_speaker_rows": 1456,
                "verifier_sha256": present.sha(source_dir / "result_verify.py"),
                "file_hashes": {name: present.sha(name) for name in paths}}
    write_json(analysis / "verification.json", approval)
    monkeypatch.setattr(present.core_verify, "verify_plan_file", lambda *a, **k: {"plan_sha256": plan["plan_sha256"]})
    monkeypatch.setattr(present.core_score, "verify_closed", lambda *a: (closure, {}))
    return repo, plan_path, run, analysis


def test_unclosed_core_precedes_scientific_read(approved_TEST, monkeypatch):
    def incomplete(*args):
        raise ValueError("TEST incomplete 720-unit core")
    monkeypatch.setattr(present.core_score, "verify_closed", incomplete)
    original, names = present.read_json, []
    def watch(path):
        names.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(present, "read_json", watch)
    monkeypatch.setattr(present, "parse_score_payload", lambda value: pytest.fail("scientific interpretation before closure"))
    with pytest.raises(ValueError, match="incomplete"):
        present.load_approved(*approved_TEST)
    assert "score.json" not in names and "verification.json" not in names


def test_unapproved_report_precedes_scientific_read(approved_TEST, monkeypatch):
    approval = approved_TEST[3] / "verification.json"
    value = present.read_json(approval)
    value["results_approved_for_paper"] = False
    write_json(approval, value)
    original, names = present.read_json, []
    def watch(path):
        names.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(present, "read_json", watch)
    monkeypatch.setattr(present, "parse_score_payload", lambda value: pytest.fail("scientific interpretation before approval"))
    with pytest.raises(ValueError, match="not approved"):
        present.load_approved(*approved_TEST)
    assert "score.json" not in names


@pytest.mark.parametrize("target", ["score.json", "per_speaker.csv", "SCORE_DONE"])
def test_modified_verified_analysis_rejected(approved_TEST, target):
    with (approved_TEST[3] / target).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="verified artifact changed"):
        present.load_approved(*approved_TEST)


def test_missing_snapshot_binding_rejected(approved_TEST):
    path = approved_TEST[3] / "verification.json"
    report = present.read_json(path)
    del report["file_hashes"][str((approved_TEST[3] / "score.json").resolve())]
    write_json(path, report)
    with pytest.raises(ValueError, match="snapshot is incomplete"):
        present.load_approved(*approved_TEST)


def test_score_replaced_after_snapshot_rejected_before_parsing(approved_TEST, monkeypatch):
    score_path = approved_TEST[3] / "score.json"
    replacement = present.read_json(score_path)
    replacement["absolute_uar"][0]["estimate_pp"] = 93.
    original_read_bytes, payload_reads = Path.read_bytes, []
    def racing_read(path):
        if path == score_path:
            payload_reads.append(path)
            write_json(path, replacement)
        return original_read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", racing_read)
    monkeypatch.setattr(present, "parse_score_payload", lambda value: pytest.fail("unapproved scientific bytes interpreted"))
    with pytest.raises(ValueError, match="scientific payload changed"):
        present.load_approved(*approved_TEST)
    assert payload_reads == [score_path]


def test_executing_source_mismatch_precedes_scientific_parsing(approved_TEST, monkeypatch):
    copied_source = approved_TEST[0] / "v3/data_design/core_present.py"
    with copied_source.open("ab") as handle:
        handle.write(b"\n# TEST source identity mismatch\n")
    monkeypatch.setattr(present, "parse_score_payload", lambda value: pytest.fail("scientific interpretation before source binding"))
    with pytest.raises(ValueError, match="executing modules differ"):
        present.load_approved(*approved_TEST)


def test_fixed_order_complete_tables_preserve_values(tmp_path):
    report = TEST_report()
    expected = {(r["model"], r["B"], r["S"], r["scenario"]): r["estimate_pp"] for r in report["absolute_uar"]}
    report["absolute_uar"].reverse()
    report["contrasts"].reverse()
    present.write_tables(report, tmp_path)
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 28 and sum(r["primary"] == "True" for r in rows) == 4
    assert [tuple((r["model"], int(r["B"]), int(r["S"]), r["scenario"])) for r in rows[:16]] == list(present.POLICIES)
    for row in rows[:16]:
        key = (row["model"], int(row["B"]), int(row["S"]), row["scenario"])
        assert float(row["estimate"]) == expected[key]
    with (tmp_path / "draw_estimates.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 28
    assert "Pointwise 95% conditional" in (tmp_path / "tables.md").read_text(encoding="utf-8")


def test_partial_scientific_table_cannot_render(tmp_path):
    report = TEST_report()
    report["contrasts"].pop()
    with pytest.raises(ValueError, match="all 16"):
        present.write_tables(report, tmp_path)


def test_png_pdf_fixture_and_immutable_reuse(approved_TEST):
    repo, plan_path, run, analysis = approved_TEST
    out = repo / "v3/data_design/work/presentation"
    first = present.present(repo, plan_path, run, analysis, out)
    assert (out / "absolute_uar.png").read_bytes().startswith(b"\x89PNG")
    assert (out / "allocation_contrasts.pdf").read_bytes().startswith(b"%PDF")
    assert present.present(repo, plan_path, run, analysis, out) == first
    with (out / "tables.md").open("ab") as handle:
        handle.write(b"\nchanged")
    with pytest.raises(ValueError, match="corrupt"):
        present.present(repo, plan_path, run, analysis, out)


def test_source_output_paths_are_forbidden(approved_TEST):
    repo, plan_path, run, analysis = approved_TEST
    for out in (repo, repo / "paper", run, analysis, repo / "v2"):
        with pytest.raises(ValueError, match="dedicated"):
            present.present(repo, plan_path, run, analysis, out)
