from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from v2_1.n14r import plan, seeds
from v2_1.n14r.spec import HPO_GRID, SPEC, sha256_json, validate_spec


def synthetic_rows(n_speakers: int = 91) -> list[dict[str, str]]:
    rows = []
    for speaker in range(n_speakers):
        for label in range(6):
            for take in range(1 + ((speaker * speaker + label * 7 + speaker * label * 3) % 5)):
                i = len(rows)
                rows.append({
                    "sample_index": str(i),
                    "relative_path": f"s{speaker:02d}_y{label}_t{take}.wav",
                    "speaker": f"s{speaker:02d}",
                    "label": f"y{label}",
                    "label_index": str(label),
                    "sha256": hashlib.sha256(f"{speaker}|{label}|{take}".encode()).hexdigest(),
                })
    return rows


def test_frozen_spec_and_grid():
    validate_spec()
    assert len(HPO_GRID) == 8
    assert HPO_GRID[4]["lr"] == 1e-3
    assert HPO_GRID[4]["weight_decay"] == 1e-4
    assert HPO_GRID[4]["dropout"] == 0.1
    assert HPO_GRID[4]["engine"] == "p1_frozen"
    assert HPO_GRID[4]["hpo_config_index"] == 4
    assert SPEC["power"] == {
        "model": "noncentral t, two-sided alpha 0.05",
        "alternative_mean_pp": 1.5,
        "sd_pp": 2.5,
        "n_draws": 24,
        "power": 0.804,
    }
    frozen = json.loads((Path(__file__).parents[1] / "spec.json").read_text(encoding="utf-8"))
    assert frozen["seed_namespace"] == "SER26-N14R"
    assert frozen["primary_inference"]["family_size"] == 1
    assert frozen["primary_inference"]["directional_success"] == (
        "mean > 0 and Holm-adjusted two-sided p <= 0.05"
    )
    assert SPEC["inference"]["directional_claim_requires"] == (
        "mean_delta>0 and holm_adjusted_two_sided_p<=0.05"
    )
    assert frozen["legacy_pilot"]["excluded_from_confirmatory_analysis"] is True


def test_seed_domains_and_pairing():
    assert len({seeds.split_seed(d) for d in range(28)}) == 28
    assert seeds.train_seed(3, 2) == seeds.train_seed(3, 2)
    assert seeds.inner_seed(3, 2, "GR_hpo") != seeds.inner_seed(3, 2, "GG_hpo")
    assert seeds.split_seed(0) != seeds.stable_u32("SER26|split|cremad|0")


def test_plan_is_complete_paired_and_scientifically_hashed():
    rows = synthetic_rows()
    units, split_docs = plan.make_plan(rows, "a" * 64)
    assert len(units) == 2240
    assert len({u["unit_id"] for u in units}) == 2240
    assert sum(u["draw_role"] == "primary" for u in units) == 1920
    assert sum(u["draw_role"] == "reserve" for u in units) == 320
    for draw_id in range(28):
        draw = [u for u in units if u["draw_id"] == draw_id]
        assert len(draw) == 80
        for fold in range(5):
            fr = [u for u in draw if u["fold"] == fold]
            assert len({u["outer_sha256"] for u in fr}) == 1
            assert len({u["train_seed"] for u in fr}) == 1
            assert {u["cell"] for u in fr} == {"GR_hpo", "GG_hpo"}
    # A scientific-field mutation must produce a different identity.
    u = units[0]
    payload_a = {k: u[k] for k in u if k not in {"unit_id", "est_gpu_sec", "status"}}
    payload_a.update(seed_namespace="SER26-N14R", config=HPO_GRID[0])
    assert u["unit_id"] == sha256_json(payload_a)
    payload_b = dict(payload_a, train_seed=u["train_seed"] + 1)
    assert sha256_json(payload_b) != u["unit_id"]
    assert all(len(doc["folds"]) == 5 for doc in split_docs.values())
    for key, doc in split_docs.items():
        expected = key.endswith("GR_hpo")
        assert all((f["meta"]["inner_overlap_speaker_count"] > 0) == expected for f in doc["folds"])


def test_build_writes_frozen_plan(tmp_path: Path):
    manifest = tmp_path / "cremad_manifest.csv"
    fields = ["sample_index", "relative_path", "speaker", "label", "label_index", "sha256"]
    import csv
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(synthetic_rows())
    out = tmp_path / "plan"
    summary = plan.build(manifest, out)
    assert summary["primary_units"] == 1920
    assert (out / "run_plan.csv").exists()
    assert len(list((out / "splits").glob("n14r__*.json"))) == 56
    assert (out / "split_index.json").exists()
    assert (out / "unit_configs.json").exists()
    hygiene = json.loads((out / "hygiene.json").read_text(encoding="utf-8"))
    assert hygiene["schema"] == "ser26-n14r-hygiene-1"
    assert set(hygiene) == {
        "schema", "manifest_sha256", "corpus", "n_input", "n_kept", "n_dropped", "dropped"
    }
    assert hygiene["n_input"] == hygiene["n_kept"] == len(synthetic_rows())
    assert hygiene["n_dropped"] == 0
    assert summary["hygiene_sha256"] == hashlib.sha256((out / "hygiene.json").read_bytes()).hexdigest()


def test_plan_rejects_missing_class():
    rows = [r for r in synthetic_rows() if r["label_index"] != "5"]
    # load_manifest performs this check for disk input; the lower-level planner
    # also fails before creating a misleading plan.
    try:
        plan.make_plan(rows, "b" * 64)
    except (AssertionError, ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("a five-class manifest was accepted")


def test_real_plan_loads_through_legacy_planio(tmp_path: Path):
    """The frozen plan must be executable by the existing v2 trainer PlanIO."""
    from v2.ser_v2.train import PlanIO

    repo = Path(__file__).resolve().parents[3]
    manifests = repo / "v2" / "manifests"
    out = tmp_path / "plan"
    plan.build(manifests / "cremad_manifest.csv", out)
    io = PlanIO(out, manifests, tmp_path / "run")
    row = io.plan[0]
    fold = io.fold(row)
    assert row["arm"] == "N14R"
    assert row["base_corpus"] == "cremad"
    assert set(("fit", "val", "test")).issubset(fold)
    assert io.configs[row["config_sha256"]]["engine"] == "p1_frozen"


EXPECTED_CREMA_DROPS = {
    "1006_TIE_HAP_XX.wav",
    "1006_TIE_NEU_XX.wav",
    "1013_WSI_DIS_XX.wav",
    "1013_WSI_SAD_XX.wav",
    "1017_IWW_ANG_XX.wav",
    "1017_IWW_FEA_XX.wav",
    "1076_MTI_SAD_XX.wav",
}


def test_real_manifest_hygiene_is_exact_and_renumbered():
    repo = Path(__file__).resolve().parents[3]
    manifest = repo / "v2" / "manifests" / "cremad_manifest.csv"
    rows, raw_sha, hygiene = plan.load_manifest(manifest)
    assert len(rows) == 7435
    assert hygiene["n_input"] == 7442
    assert hygiene["n_kept"] == 7435
    assert hygiene["n_dropped"] == 7
    assert hygiene["manifest_sha256"] == raw_sha
    assert {item["relative_path"] for item in hygiene["dropped"]} == EXPECTED_CREMA_DROPS
    assert all(item["reason"].startswith(("H1 ", "H3 ")) for item in hygiene["dropped"])
    assert [int(row["sample_index"]) for row in rows] == list(range(7435))
    assert not (EXPECTED_CREMA_DROPS & {row["relative_path"] for row in rows})
    assert raw_sha == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_real_splits_exclude_drops_and_match_planio_and_pinned_cache(tmp_path: Path):
    from v2.ser_v2.train import PlanIO

    repo = Path(__file__).resolve().parents[3]
    manifests = repo / "v2" / "manifests"
    out = tmp_path / "plan"
    summary = plan.build(manifests / "cremad_manifest.csv", out)
    assert summary["manifest_input_rows"] == 7442
    assert summary["analysis_population_rows"] == 7435
    io = PlanIO(out, manifests, tmp_path / "run")
    cache_dir = Path(r"E:\科研\essay\SER-v2\v2\features")
    candidates = sorted(cache_dir.glob("cremad__logmel__*.npz"))
    if len(candidates) != 1:
        pytest.skip("the pinned external CREMA-D log-mel cache is unavailable")
    with np.load(candidates[0], allow_pickle=False) as archive:
        cache_paths = [str(value) for value in archive["paths"].tolist()]
    clean_rows, _, _ = plan.load_manifest(manifests / "cremad_manifest.csv")
    clean_paths = [row["relative_path"] for row in clean_rows]
    assert cache_paths == clean_paths
    assert set(io.manifest["cremad"]) == set(cache_paths)

    split_index = json.loads((out / "split_index.json").read_text(encoding="utf-8"))
    for entry in split_index.values():
        table = json.loads((out / entry["path"]).read_text(encoding="utf-8"))
        assert table["population"] == cache_paths
        assert not (EXPECTED_CREMA_DROPS & set(table["population"]))
        for fold in table["folds"]:
            assert set(fold["fit"] + fold["val"] + fold["test"]) == set(cache_paths)
