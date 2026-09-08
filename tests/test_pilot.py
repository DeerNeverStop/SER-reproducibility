"""Small generated TEST fixtures only; no production feature cache or SER score read."""
import csv
import json
from pathlib import Path

import pytest

from v3.data_design import pilot


@pytest.fixture
def fixture(tmp_path):
    import numpy as np
    paths = [f"s{s}_p0_c{c}.wav" for s in range(3) for c in range(6)]
    rows = [{"relative_path": p, "speaker": p.split("_")[0], "corpus": "cremad",
             "sha256": pilot.digest(p), "label_index": str(i % 6)} for i, p in enumerate(paths)]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    features = tmp_path / "cremad__wavlm_base_plus.npz"
    np.savez(features, paths=np.asarray(paths), X=np.random.default_rng(7).normal(size=(18, 13, 768)).astype("float32"))
    meta = {"encoder": "wavlm_base_plus", "corpus": "cremad", "sha256": pilot.file_sha(features), "n": 18}
    pilot.atomic(features.with_suffix(".json"), meta)
    plan = {"schema": pilot.SCHEMA,
            "input": {"manifest_path": "manifest.csv", "manifest_sha256": pilot.file_sha(manifest),
                      "feature_sha256": pilot.file_sha(features), "feature_kind": "wavlm_base_plus", "feature_state": 12},
            "units": [{"unit_id": "unit_01", "fit": paths[:12], "select": paths[12:],
                       "config": {"B": 12, "S": 2, "P_global": 1}, "train_seed": 3}]}
    return tmp_path, plan, features


def save_plan(root, plan):
    plan["plan_sha256"] = pilot.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    path = root / "plan.json"
    pilot.atomic(path, plan)
    return path


def execute(fixture):
    root, plan, features = fixture
    return pilot.run(root, save_plan(root, plan), features, root / "out")


def test_blind_run_and_verified_resume(fixture):
    import numpy as np
    first = execute(fixture)
    assert first["completed_units"] == first["fitted_this_invocation"] == 1
    root = fixture[0]
    with np.load(root / "out/units/unit_01/predictions.npz", allow_pickle=False) as z:
        assert set(z.files) == {"paths", "logits"}
        assert z["logits"].shape == (6, 6)
    second = execute(fixture)
    assert second["reused_units"] == 1 and second["fitted_this_invocation"] == 0
    receipt = json.dumps(pilot.read_json(root / "out/units/unit_01/unit.json"))
    assert all(token not in receipt.lower() for token in ("uar", "accuracy", "y_true", "rank"))


@pytest.mark.parametrize("corruption,match", [("speaker", "speaker leakage"), ("bytes", "byte duplicate"),
                                             ("cache", "feature hash mismatch"), ("test", "test inputs forbidden"),
                                             ("class", "fit missing fixed class"), ("duplicate", "duplicate panel paths")])
def test_fail_closed_inputs(fixture, corruption, match):
    root, plan, features = fixture
    unit = plan["units"][0]
    if corruption == "cache":
        with features.open("ab") as f:
            f.write(b"changed")
    elif corruption == "test":
        unit["test"] = unit["select"]
    elif corruption == "duplicate":
        unit["fit"][0] = unit["fit"][1]
    elif corruption == "class":
        unit["fit"] = [p for p in unit["fit"] if "c5" not in p]
    else:
        manifest = root / "manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        key = "speaker" if corruption == "speaker" else "sha256"
        rows[-1][key] = rows[0][key]
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plan["input"]["manifest_sha256"] = pilot.file_sha(manifest)
    with pytest.raises(ValueError, match=match):
        execute(fixture)


@pytest.mark.parametrize("target", ["DONE", "unit.json", "predictions.npz"])
def test_resume_rejects_tampering(fixture, target):
    execute(fixture)
    path = fixture[0] / "out/units/unit_01" / target
    if target == "DONE":
        data = pilot.read_json(path)
        data["receipt_sha256"] = "0" * 64
        pilot.atomic(path, data)
    else:
        with path.open("ab") as f:
            f.write(b" ")
    with pytest.raises(ValueError, match="DONE hash mismatch"):
        execute(fixture)


def test_changed_plan_cannot_resume(fixture):
    execute(fixture)
    fixture[1]["units"][0]["train_seed"] = 8
    with pytest.raises(ValueError, match="different plan"):
        execute(fixture)


def test_selection_distribution_cannot_change_other_predictions(fixture):
    """Changing held-out peers must not change one untouched held-out prediction."""
    import numpy as np
    root, plan, features = fixture
    execute(fixture)
    with np.load(root / "out/units/unit_01/predictions.npz", allow_pickle=False) as z:
        before = z["logits"][0].copy()
    with np.load(features, allow_pickle=False) as z:
        X, paths = z["X"], z["paths"]
    X[13:] = X[13:] * 1e6 + 1e9
    np.savez(features, X=X, paths=paths)
    meta = pilot.read_json(features.with_suffix(".json"))
    meta["sha256"] = plan["input"]["feature_sha256"] = pilot.file_sha(features)
    pilot.atomic(features.with_suffix(".json"), meta)
    pilot.run(root, save_plan(root, plan), features, root / "out_perturbed")
    with np.load(root / "out_perturbed/units/unit_01/predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(before, z["logits"][0])


def test_interrupted_unit_is_rebuilt_before_completion(fixture):
    execute(fixture)
    folder = fixture[0] / "out/units/unit_01"
    (folder / "DONE").unlink()
    (folder / "predictions.npz").write_bytes(b"interrupted atomic-write TEST fixture")
    completion = execute(fixture)
    assert completion["fitted_this_invocation"] == 1 and completion["reused_units"] == 0


@pytest.mark.parametrize("corruption", ["encoder", "row_count", "paths", "shape", "nonfinite"])
def test_feature_column_and_metadata_identity(fixture, corruption):
    import numpy as np
    root, plan, features = fixture
    metadata = pilot.read_json(features.with_suffix(".json"))
    if corruption in ("encoder", "row_count"):
        metadata["encoder" if corruption == "encoder" else "n"] = "hubert_base" if corruption == "encoder" else 19
    else:
        with np.load(features, allow_pickle=False) as z:
            X, paths = z["X"], z["paths"]
        if corruption == "paths":
            paths[0] = paths[1]
        elif corruption == "shape":
            X = X[:, :12]
        else:
            X[0, 0, 0] = np.nan
        np.savez(features, X=X, paths=paths)
        metadata["sha256"] = plan["input"]["feature_sha256"] = pilot.file_sha(features)
    pilot.atomic(features.with_suffix(".json"), metadata)
    with pytest.raises(ValueError):
        execute(fixture)
