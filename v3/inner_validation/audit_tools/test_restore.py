"""Small synthetic files only; never imports or runs a real GPU model."""
import argparse
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from v3.inner_validation.audit_tools import restore


def units_fixture():
    roles = {r: [f"SYNTHETIC_{r}/{i}.wav" for i in range(6)] for r in restore.ROLES}
    units = [{"unit_id": uid, "draw": 0, "fold": 0, "config_index": i, **copy.deepcopy(roles)}
             for i, uid in enumerate(restore.FIXED_IDS)]
    return {"units": units + [{"unit_id": "unselected", "draw": 1, "fold": 0}]}


def prediction_file(path, unit):
    arrays = {}
    for role in restore.ROLES:
        arrays[role + "__paths"] = np.asarray(unit[role])
        arrays[role + "__labels"] = np.arange(6, dtype=np.int64)
        for i, checkpoint in enumerate(restore.CHECKPOINTS):
            arrays[f"{role}__{checkpoint}__logits"] = np.full((6, 6), i, dtype=np.float64)
    np.savez(path, **arrays)
    return arrays


def test_fixed_selection_cannot_expand_or_change_config_or_role_order():
    plan = units_fixture()
    assert [u["unit_id"] for u in restore.select_units(plan)] == restore.FIXED_IDS
    missing = copy.deepcopy(plan)
    missing["units"].pop(0)
    with pytest.raises(ValueError, match="panel missing"):
        restore.select_units(missing)
    reordered = copy.deepcopy(plan)
    reordered["units"][1]["test"].reverse()
    with pytest.raises(ValueError, match="evaluation population/order"):
        restore.select_units(reordered)
    changed = copy.deepcopy(plan)
    changed["units"][0]["unit_id"] = "outcome_selected_other_unit"
    with pytest.raises(ValueError, match="identities differ"):
        restore.select_units(changed)


def test_expected_npz_preserves_all_nine_checkpoint_roles_and_rejects_corruption(tmp_path):
    unit = units_fixture()["units"][0]
    path = tmp_path / "synthetic.npz"
    arrays = prediction_file(path, unit)
    expected = restore.load_expected(path, unit)
    assert len(expected) == 9
    assert np.all(expected["val_seen__best_unseen__logits"] == 1)
    assert np.all(expected["test__last__logits"] == 2)
    arrays["val_unseen__paths"] = arrays["val_unseen__paths"][::-1]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="path order"):
        restore.load_expected(path, unit)
    arrays = prediction_file(path, unit)
    arrays["test__last__logits"][0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="invalid stored logits"):
        restore.load_expected(path, unit)


def test_compare_tolerance_is_not_accuracy_and_shape_cannot_broadcast():
    expected = np.arange(36, dtype=np.float64).reshape(6, 6)
    assert restore.compare(expected, expected.copy())["max_abs_diff"] == 0.
    changed = expected.copy()
    changed[0, 0] += .001  # Argmax is unchanged, yet numerical restoration fails.
    result = restore.compare(expected, changed)
    assert result["pass"] is False and result["max_abs_diff"] == pytest.approx(.001)
    assert set(result) == {"pass", "max_abs_diff", "rows", "classes", "rtol", "atol"}
    with pytest.raises(ValueError, match="invalid restored"):
        restore.compare(expected, np.zeros((1, 6)))


def test_existing_output_refused_before_loading_models_or_inputs(tmp_path):
    out = tmp_path / "existing_report"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("UNCHANGED")
    args = argparse.Namespace(repo=tmp_path / "missing_repo", outroot=tmp_path / "missing_results",
                              plan=tmp_path / "missing_plan", model=tmp_path / "missing_model",
                              audio_root=tmp_path / "missing_audio", out=out)
    with pytest.raises(ValueError, match="repeated/existing"):
        restore.execute(args)
    assert sentinel.read_text() == "UNCHANGED"
    assert list(out.iterdir()) == [sentinel]


def test_wrong_checkout_or_changed_environment_rejected(tmp_path):
    repo = tmp_path / "repo"
    right = SimpleNamespace(__file__=str(repo / "v3/inner_validation/engine.py"))
    restore.check_module(right, repo, "v3/inner_validation/engine.py")
    wrong = SimpleNamespace(__file__=str(tmp_path / "other/engine.py"))
    with pytest.raises(ValueError, match="pinned checkout"):
        restore.check_module(wrong, repo, "v3/inner_validation/engine.py")
    env = {key: "SYNTHETIC" for key in ("python", "numpy", "torch", "cuda", "cudnn", "gpu",
                                         "threads", "matmul_tf32", "cudnn_tf32")}
    restore.check_environment(env, env)
    changed = dict(env, cudnn_tf32=False)
    with pytest.raises(ValueError, match="cudnn_tf32"):
        restore.check_environment(env, changed)
