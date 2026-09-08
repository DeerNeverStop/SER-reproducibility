"""Post-hoc reconciliation of descriptors D04, D07 and D08 between the scorer (results.json) and the
verifier (verifier_results.json). The two frozen implementations chose different key names for these
three descriptors, so `ser_v2.verify` compared none of their values (verification.json holds D01, D02,
D03, D05, D06, D09 and D14 only). This tool maps the keys and compares the numbers; it never edits
either file.

    python -m tools.reconcile_descriptors --run runs/main --out runs/main/descriptors/reconcile_d04_d07_d08.json
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import atomic_write_json, read_json  # noqa: E402

TOL = 1e-9


def cmp(items, name, a, b):
    if a is None and b is None:
        items.append({"item": name, "scorer": None, "verifier": None, "abs_diff": 0.0, "pass": True})
        return
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        items.append({"item": name, "scorer": a, "verifier": b, "abs_diff": None, "pass": False})
        return
    d = abs(float(a) - float(b))
    items.append({"item": name, "scorer": a, "verifier": b, "abs_diff": d, "pass": d <= TOL})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    S = read_json(a.run / "results.json")["descriptors"]
    V = read_json(a.run / "verifier_results.json")["descriptors"]
    items = []
    # D04
    for sk, vk in (("sub_1400_one_minus_sub_700_one", "sub_1400_one_vs_sub_700_one"), ("sub_700_two_minus_sub_700_one", "sub_700_two_vs_sub_700_one")):
        s, v = S["D04"][sk], V["D04"][vk]
        for f in ("n", "mean", "ci_low", "ci_high"):
            cmp(items, f"D04.{sk}.{f}", s.get(f), v.get(f))
    # D07
    for fam, vpre, vsd in (("cremad_24", "p24", "sd_p24"), ("cremad_91m", "p91", "sd_p91")):
        for k, val in S["D07"][fam].items():
            if k.startswith("_"):
                cmp(items, f"D07.{fam}.sd_over_draws", val, V["D07"].get(vsd))
            else:
                d = k.split("_d")[-1]
                cmp(items, f"D07.{fam}.{k}", val, V["D07"]["draws"].get(f"d{d}", {}).get(f"{vpre}_mean"))
    # D08
    s, v = S["D08"]["rr_minus_gg"], V["D08"]["premium_RR_hpo_minus_GG_hpo"]
    for f in ("n", "mean", "ci_low", "ci_high"):
        cmp(items, f"D08.rr_minus_gg.{f}", s.get(f), v.get(f))
    for cell in ("RR_hpo", "GR_hpo", "GG_hpo"):
        cmp(items, f"D08.{cell}.test_selection_optimism_mean", S["D08"].get(cell, {}).get("test_selection_optimism_mean"), (V["D08"]["test_selection_optimism"].get(cell) or {}).get("mean"))
        ssel = S["D08"].get(cell, {}).get("selected", {}); vsel = V["D08"]["selected_config_index"].get(cell) or {}
        for f, idx in ssel.items():
            cmp(items, f"D08.{cell}.selected.f{f}", idx, vsel.get(f"f{f}"))
    diffs = [x["abs_diff"] for x in items if x["abs_diff"] is not None]
    rep = {"tolerance": TOL, "n_items": len(items), "n_failed": sum(1 for x in items if not x["pass"]), "max_abs_diff": max(diffs) if diffs else None,
           "pass": all(x["pass"] for x in items), "items": items,
           "note": "key map between ser_v2.score (results.json) and ser_v2.verify (verifier_results.json) for the three descriptors the frozen verifier did not compare"}
    atomic_write_json(a.out, rep)
    print({k: v for k, v in rep.items() if k not in ("items", "note")})
    for x in items:
        if not x["pass"]:
            print("FAIL", x)


if __name__ == "__main__":
    main()
