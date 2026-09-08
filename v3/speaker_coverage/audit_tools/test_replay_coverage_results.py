"""Synthetic-only tests for the external numerical replay; no real E2 files."""
from copy import deepcopy
import csv
import json
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import replay_coverage_results as r


def test_boolean_recall_pools_actual_records_before_macro_average():
    small_perfect = [(label, label) for label in range(6)]
    large_wrong = [(label, (label + 1) % 6) for label in range(6) for _ in range(3)]
    assert r.boolean_recalls(small_perfect + large_wrong) == (25.0, 4)
    assert (r.boolean_recalls(small_perfect)[0] + r.boolean_recalls(large_wrong)[0]) / 2 == 50


def test_missing_class_is_not_dropped_from_macro_denominator():
    with pytest.raises(ValueError, match="class is absent"):
        r.boolean_recalls([(label, label) for label in range(5)])


def test_bootstrap_multiplicity_matches_same_fixed_stream_at_odd_people_count():
    n, repetitions = 91, 10000
    weights = r.bootstrap_people_weights(n, repetitions)
    indexes = np.random.default_rng(20260906).integers(0, n, size=(repetitions, n))
    differences = np.linspace(-17.9, 32.4, n) ** 3 / 1000
    weighted = weights @ differences / n
    explicit = np.asarray([sum(differences[j] for j in row) / n for row in indexes])
    np.testing.assert_allclose(weighted, explicit, atol=1e-12, rtol=0)
    for quantile in (.025, .25, .975):
        assert abs(r.linear_quantile(weighted, quantile) - float(np.quantile(explicit, quantile, method="linear"))) < 1e-12


@pytest.fixture(scope="module")
def synthetic_queries(tmp_path_factory):
    """720 tiny synthetic artifacts exercise the actual complete factor grid."""
    directory = tmp_path_factory.mktemp("replay-synthetic-only")
    speakers = [f"fixture{i}" for i in range(10)]
    metadata, units, files, expected = {}, [], {}, {}
    for model in r.MODELS:
        for policy, repeat, fold, rotation in product(r.POLICIES, range(r.repeats(model)), range(5), range(6)):
            people = [s for i, s in enumerate(speakers) if i % 5 == fold]
            paths, truth = [], []
            for person in people:
                for label in range(6):
                    for copy in range(rotation + 1):
                        name = f"{person}-rotation{rotation}-class{label}-copy{copy}.wav"
                        metadata[name] = {"speaker": person, "label_index": str(label)}
                        paths.append(name)
                        truth.append(label)
            uid = r.digest([model, policy, repeat, fold, rotation])
            unit = {"unit_id": uid, "model": model, "policy": policy, "fold": fold, "rotation": rotation,
                    "block": "ft" if model == "wavlm_ft" else "core",
                    "draw": 0 if model == "wavlm_ft" else repeat, "seed_index": repeat if model == "wavlm_ft" else 0,
                    "test": paths}
            arrays = {"paths": np.asarray(paths), "labels": np.asarray(truth, dtype=np.int64)}
            for checkpoint, prefix in (("best", ""), ("last", "last_")):
                threshold = {"U": 0, "R": 1, "C": 2}[policy] + repeat % 2 - (checkpoint == "last")
                guesses = [label if rotation <= threshold else (label + 1) % 6 for label in truth]
                logits = np.full((len(paths), 6), -2.0, dtype=np.float64)
                logits[np.arange(len(paths)), guesses] = 2.0
                ex = np.exp(logits - logits.max(1, keepdims=True))
                arrays.update({prefix + "logits": logits, prefix + "pred": np.asarray(guesses, dtype=np.int64),
                               prefix + "proba": ex / ex.sum(1, keepdims=True)})
                total_correct = sum(range(1, max(0, threshold + 2)))
                for person in people:
                    expected[model, policy, checkpoint, repeat, person] = total_correct / 21 * 100
            path = directory / (uid + ".npz")
            np.savez(path, **arrays)
            files[uid] = path, r.file_hash(path)
            units.append(unit)
    assert len(units) == 720
    return {"units": units}, files, metadata, speakers, expected


def test_complete_factor_grid_uses_six_rotations_and_ft_two_seeds(synthetic_queries):
    plan, files, metadata, speakers, expected = synthetic_queries
    values, detail, folds = r.collect_observations(plan, files, metadata, speakers)
    assert values.keys() == expected.keys() and len(detail) == 480
    assert all(abs(values[k] - expected[k]) < 1e-12 for k in values)
    assert all(row["query_records"] == 126 and row["minimum_class_support"] == 21 for row in detail)
    assert {row["repeat"] for row in detail if row["model"] == "wavlm_ft"} == {0, 1}
    tables = r.summary_from_person_values(speakers, values, folds)
    assert len(tables["contrasts.csv"]) == 12 and len(tables["policy_means.csv"]) == 18
    assert len(tables["per_speaker.csv"]) == 180 and len(tables["fold_contrasts.csv"]) == 60
    for contrast in tables["contrasts.csv"]:
        # Every synthetic person has the same policy effect. People bootstrap
        # must therefore collapse exactly to that effect, regardless of draw.
        assert abs(contrast["ci95_low_pp"] - contrast["difference_pp"]) < 1e-12
        assert abs(contrast["ci95_high_pp"] - contrast["difference_pp"]) < 1e-12


def test_duplicate_rotation_cannot_be_counted_as_an_extra_repetition(synthetic_queries):
    plan, files, metadata, speakers, _ = synthetic_queries
    duplicated = {"units": plan["units"] + [deepcopy(plan["units"][0])]}
    with pytest.raises(ValueError, match="repeated query recording/rotation"):
        r.collect_observations(duplicated, files, metadata, speakers)


def test_unseen_recording_labels_and_saved_argmax_are_independently_checked(tmp_path):
    path = tmp_path / "tiny.npz"
    unit = {"test": ["a.wav", "b.wav"]}
    metadata = {"a.wav": {"label_index": "0"}, "b.wav": {"label_index": "1"}}
    logits = np.array([[2., 2., 0., 0., 0., 0.], [0., 3., 0., 0., 0., 0.]])
    arrays = {"paths": np.array(unit["test"]), "labels": np.array([0, 1], dtype=np.int64)}
    for prefix in ("", "last_"):
        arrays.update({prefix + "logits": logits, prefix + "pred": np.array([0, 1], dtype=np.int64),
                       prefix + "proba": np.zeros((2, 6))})
    np.savez(path, **arrays)
    assert r.read_guesses(path, unit, metadata)[2]["best"] == [0, 1]
    arrays["last_pred"] = np.array([1, 1], dtype=np.int64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="independent argmax"):
        r.read_guesses(path, unit, metadata)
    arrays["last_pred"] = np.array([0, 1], dtype=np.int64)
    arrays["labels"] = np.array([1, 1], dtype=np.int64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="raw manifest"):
        r.read_guesses(path, unit, metadata)


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update(value="nan"), "invalid numeric"),
    (lambda rows: rows[0].update(value="51.0"), "numeric disagreement"),
    (lambda rows: rows.append(deepcopy(rows[0])), "row count"),
    (lambda rows: rows[0].update(speaker="other"), "unknown/duplicate"),
])
def test_table_resealing_does_not_hide_numeric_or_population_changes(mutation, match):
    expected = [{"speaker": "a", "value": 50.0, "n": 3, "per_repeat": [40., 50., 60.]}]
    rows = [{"speaker": "a", "value": "50.0", "n": "3", "per_repeat": "[40,50,60]"}]
    r.Comparator().rows(rows, expected, ("speaker",), "fixture.csv", csv_input=True)
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        r.Comparator().rows(rows, expected, ("speaker",), "fixture.csv", csv_input=True)


def test_duplicate_json_key_is_rejected_instead_of_silently_overwritten():
    with pytest.raises(ValueError, match="duplicate JSON"):
        r.parse_json('{"pass":false,"pass":true}')


def test_missing_formal_inputs_write_failure_without_reading_pilot(tmp_path, capsys):
    paths = {key: tmp_path / key for key in ("repo", "plan", "outroot", "results")}
    target = tmp_path / "audit.json"
    argv = [part for key, path in paths.items() for part in ("--" + key, str(path))] + ["--out", str(target)]
    assert r.main(argv) == 1
    report = json.loads(target.read_text())
    assert report["pass"] is False and report["new_model_fits"] == 0 and report["phase"] == "formal"
    assert "FileNotFoundError" == report["error_type"]
    assert json.loads(capsys.readouterr().out)["pass"] is False


@pytest.fixture
def synthetic_main_report(tmp_path, synthetic_queries):
    plan, files, metadata, speakers, _ = synthetic_queries
    values, detail, folds = r.collect_observations(plan, files, metadata, speakers)
    tables = r.summary_from_person_values(speakers, values, folds)
    tables["per_repeat_speaker.csv"] = detail
    plan = {**plan, "plan_sha256": "a" * 64}
    closure = {u["unit_id"]: r.digest(["synthetic DONE", u["unit_id"]]) for u in plan["units"]}
    results, repo = tmp_path / "results", tmp_path / "repo"
    results.mkdir()
    (repo / "v3/speaker_coverage").mkdir(parents=True)
    source = repo / "v3/speaker_coverage/score.py"
    source.write_text("# Synthetic source identity only; no imported scoring implementation.\n")
    for filename, rows in tables.items():
        with (results / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    report = {"schema": "ser-speaker-coverage-analysis-1", "plan_sha256": plan["plan_sha256"],
              "done_mapping_sha256": r.digest(closure), "source_sha256": r.file_hash(source),
              "verified_formal_units": 720, "excluded_pilot_units": 19, "bootstrap_seed": r.SEED, "bootstrap_repetitions": r.BOOTSTRAPS,
              "all_comparisons_reported": True, "equivalence_claim": False, "causal_mediation_claim": False,
              "policies": deepcopy(tables["policy_means.csv"]), "contrasts": deepcopy(tables["contrasts.csv"]),
              "primary": deepcopy(next(row for row in tables["contrasts.csv"] if row["role"] == "primary")),
              "table_sha256": {filename: r.file_hash(results / filename) for filename in tables},
              "result_audits": {block: {"pass": True, "count": count, "phase": "formal", "plan_sha256": plan["plan_sha256"],
                                        "done_sha256": {u["unit_id"]: closure[u["unit_id"]] for u in plan["units"] if u["block"] == block}}
                                for block, count in (("core", 540), ("ft", 180))}}
    return results, report, tables, closure, plan, repo


def test_all_five_csvs_and_results_json_are_numerically_rechecked(synthetic_main_report):
    comparison = r.compare_score_artifacts(*synthetic_main_report)
    assert comparison.numeric_values_checked > 2000 and comparison.maximum_absolute_error == 0


def test_resealed_csv_number_does_not_pass_content_replay(synthetic_main_report):
    results, report, *_ = synthetic_main_report
    path = results / "fold_contrasts.csv"
    rows = r.read_csv(path)
    rows[0]["difference_pp_descriptive"] = str(float(rows[0]["difference_pp_descriptive"]) + 0.01)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["table_sha256"][path.name] = r.file_hash(path)
    with pytest.raises(ValueError, match="numeric disagreement.*fold_contrasts"):
        r.compare_score_artifacts(*synthetic_main_report)


@pytest.mark.parametrize("mutation,match", [
    (lambda report: report["primary"].update(ci95_low_pp=-100.0), "numeric disagreement.*primary"),
    (lambda report: report.update(done_mapping_sha256="e" * 64), "actual formal DONE closure"),
    (lambda report: report["result_audits"]["ft"].update(count=179), "block audit"),
    (lambda report: report.update(bootstrap_seed=1), "bootstrap protocol"),
])
def test_results_json_cannot_claim_another_closure_or_inference(synthetic_main_report, mutation, match):
    mutation(synthetic_main_report[1])
    with pytest.raises(ValueError, match=match):
        r.compare_score_artifacts(*synthetic_main_report)


@pytest.mark.parametrize("other_checkout,match", [(True, "another checkout"), (False, "pinned SHA256")])
def test_cached_or_unpinned_verifier_cannot_be_used_for_replay(tmp_path, monkeypatch, other_checkout, match):
    repo, results = tmp_path / "repo", tmp_path / "results"
    source = repo / "v3/speaker_coverage/verify.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic identity fixture only\n")
    results.mkdir()
    plan = {"plan_sha256": "a" * 64, "source_sha256": {"v3/speaker_coverage/verify.py": "0" * 64}}
    (results / "results.json").write_text(json.dumps({"plan_sha256": plan["plan_sha256"], "done_mapping_sha256": r.digest({})}))
    monkeypatch.setattr(r, "load_plan", lambda _: plan)
    monkeypatch.setattr(r, "load_raw_manifest", lambda *_: ({}, []))
    monkeypatch.setattr(r, "inspect_closure", lambda *_: ({}, {}))
    monkeypatch.setattr(r.sys, "path", list(r.sys.path))
    def forbidden(*args, **kwargs):
        raise AssertionError("wrong verifier or prediction collector must not execute")
    verifier = SimpleNamespace(__file__=str(tmp_path / "different/verify.py" if other_checkout else source), verify_results=forbidden)
    monkeypatch.setattr(r.importlib, "import_module", lambda _: verifier)
    monkeypatch.setattr(r, "collect_observations", forbidden)
    with pytest.raises(ValueError, match=match):
        r.replay(repo, tmp_path / "plan.json", tmp_path / "outroot", results)
