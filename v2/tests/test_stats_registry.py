import numpy as np
from pathlib import Path

from ser_v2 import registry, stats


def test_frame_envelope_reproduces_the_2026_paper():
    env = stats.frame_envelope(286, 60, 44, 7)
    assert (env["K_low"], env["K_high"]) == (177, 263)
    assert abs(env["pct_low"] - 61.888) < 0.01 and abs(env["pct_high"] - 91.958) < 0.01
    assert stats.partial_identification(44, 7, 60) == (44 / 60, 51 / 60)


def test_holm_step_down_hand_check():
    res = stats.holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert res["a"]["reject"] and not res["b"]["reject"] and not res["c"]["reject"]
    assert abs(res["a"]["p_holm"] - 0.004) < 1e-12 and abs(res["b"]["p_holm"] - 0.06) < 1e-12
    res2 = stats.holm({"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.5})
    assert res2["a"]["reject"] and res2["b"]["reject"] and res2["c"]["reject"] and not res2["d"]["reject"]


def test_uar_skips_absent_classes():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 1])
    assert abs(stats.uar(y, p, 4) - 75.0) < 1e-12


def test_paired_test_and_sign_rule():
    d = np.array([0.0] * 5 + [1.0, 2.0, 3.0, -1.0, 2.5])
    t = stats.paired_test(d)
    assert t["sign_test_used"] and t["n_nonzero"] == 5
    assert stats.paired_test(np.zeros(10))["p"] == 1.0


def test_bootstrap_is_deterministic():
    d = np.random.RandomState(0).normal(1, 2, 30)
    assert stats.speaker_bootstrap_ci(d, 2000, 20260903) == stats.speaker_bootstrap_ci(d, 2000, 20260903)


def test_tost():
    d = np.random.RandomState(1).normal(0.0, 1.0, 91)
    assert stats.tost_paired(d, 1.0)["equivalent"]
    assert not stats.tost_paired(d + 2.0, 1.0)["equivalent"]


def test_registry_is_consistent(tmp_path: Path):
    import json
    pt = json.load(open(Path(__file__).resolve().parents[1] / "registry" / "power_table.json", encoding="utf-8"))
    info = registry.export(tmp_path, pt)
    sizes = info["family_sizes"]
    assert sizes["R"] == 11 and sizes["T"] == 2 and sizes["N-decomp"] == 4
    assert all(h["a_priori_status"] in ("confirmatory", "estimate") for h in registry.HYPOTHESES)
    demoted = {h["hypothesis_id"] for h in registry.HYPOTHESES if h["a_priori_status"] == "estimate"}
    assert demoted and demoted.isdisjoint({"N01", "R01"})
    ids = {h["hypothesis_id"] for h in registry.HYPOTHESES}
    assert len(ids) == 28
    text = (tmp_path / "hypothesis_registry.csv").read_text(encoding="utf-8")
    assert "20260903" in text and "family_size" in text
