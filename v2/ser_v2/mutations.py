"""Mutation battery: inject one fault at a time into a copy of a synthetic run and check that
the scorer rejects it (integrity failure, void cell-run, `not tested` verdict) or behaves as
specified (directional reading, crossing isolation, conditional truncation).

    python -m ser_v2.mutations --plan PLAN --manifests MANIFESTS --run RUN --registry REGISTRY --work WORKDIR [--full]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np

from . import fixtures
from .common import atomic_write_json, atomic_write_text, read_csv, read_json, sha256_text
from .score import Run, Scorer


# ----------------------------------------------------------------------------- helpers

def link_tree(src: Path, dst: Path) -> None:
    """Hard-link copy (same filesystem) so a mutation only rewrites the touched files."""
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s, d = Path(root) / f, dst / rel / f
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def rewrite(path: Path, text: str) -> None:
    """Break the hard link before writing."""
    if path.exists():
        path.unlink()
    atomic_write_text(path, text)


def rewrite_json(path: Path, value) -> None:
    if path.exists():
        path.unlink()
    atomic_write_json(path, value)


def pick_units(plan: list[dict], **crit) -> list[dict]:
    out = []
    for u in plan:
        if all(str(u.get(k)) == str(v) for k, v in crit.items()):
            out.append(u)
    return out


def rehash_unit(run: Path, unit_id: str) -> None:
    udir = run / "units" / unit_id
    text = (udir / "predictions.csv").read_text(encoding="utf-8")
    sha = sha256_text(text)
    uj = read_json(udir / "unit.json")
    uj["predictions_sha256"] = sha
    rewrite_json(udir / "unit.json", uj)
    rewrite(udir / "DONE", sha + "\n")


def pred_lines(run: Path, unit_id: str) -> list[str]:
    return (run / "units" / unit_id / "predictions.csv").read_text(encoding="utf-8").splitlines()


def write_pred(run: Path, unit_id: str, lines: list[str], rehash: bool = True) -> None:
    rewrite(run / "units" / unit_id / "predictions.csv", "\n".join(lines) + "\n")
    if rehash:
        rehash_unit(run, unit_id)


def score(plan: Path, manifests: Path, run: Path, registry: Path) -> dict:
    r = Run(plan, manifests, run, registry)
    return Scorer(r).run_all()


# ----------------------------------------------------------------------------- cases

def case_drop_row(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="cnn", cell="RR", r=0, seed_index=0, fold=0)[0]
    lines = pred_lines(run, u["unit_id"])
    write_pred(run, u["unit_id"], lines[:-1], rehash=False)
    return {"integrity_fail": ["I1"]}


def case_label_swap(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="cnn", cell="RR", r=0, seed_index=0, fold=0)[0]
    lines = pred_lines(run, u["unit_id"])
    parts = lines[1].split(","); parts[3] = str((int(parts[3]) + 1) % 8)
    lines[1] = ",".join(parts)
    write_pred(run, u["unit_id"], lines)
    return {"integrity_fail": ["I3"]}


def case_fold_overlap(plan, run, ctx):
    us = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="cnn", cell="RR", r=0, seed_index=0)
    u0 = next(u for u in us if u["fold"] == "0"); u1 = next(u for u in us if u["fold"] == "1")
    l0, l1 = pred_lines(run, u0["unit_id"]), pred_lines(run, u1["unit_id"])
    write_pred(run, u0["unit_id"], l0 + [l1[1]])
    return {"integrity_fail": ["I2", "I5"]}


def case_config_drift(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="cnn", cell="GG", r=1, seed_index=1, fold=2)[0]
    uj = read_json(run / "units" / u["unit_id"] / "unit.json")
    uj["config"]["lr"] = 5e-4
    rewrite_json(run / "units" / u["unit_id"] / "unit.json", uj)
    return {"integrity_fail": ["I7"]}


def case_duplicate_row(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="resnet_se", cell="RG", r=0, seed_index=0, fold=0)[0]
    lines = pred_lines(run, u["unit_id"])
    write_pred(run, u["unit_id"], lines + [lines[1]])
    return {"integrity_fail": ["I4"]}


def case_missing_cell(plan, run, ctx):
    for u in pick_units(ctx["plan"], arm="CTRL", corpus_level="subesco_980", cell="GG"):
        if u["model"] in ("hubert_base", "wavlm_base_plus", "wav2vec2_base"):
            d = run / "units" / u["unit_id"] / "DONE"
            if d.exists():
                d.unlink()
    return {"verdict": {"N02": "not tested"}, "family_size": {"N02": ctx["family_size"]("N02")}, "unchanged": ["N01"]}


def case_done_tamper(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="PROBECPU", corpus_level="ravdess", cell="RO", r=0, fold=0)[0]
    rewrite(run / "units" / u["unit_id"] / "DONE", "0" * 64 + "\n")
    return {"integrity_fail": ["I1"]}


def case_argmax_violation(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="transformer", cell="GR", r=2, seed_index=2, fold=4)[0]
    lines = pred_lines(run, u["unit_id"])
    parts = lines[1].split(","); logits = [float(x) for x in parts[5:]]
    parts[4] = str(int(np.argmin(logits)))
    lines[1] = ",".join(parts)
    write_pred(run, u["unit_id"], lines)
    return {"integrity_fail": ["I4"]}


def case_split_table_tamper(plan, run, ctx):
    idx = read_json(plan / "split_index.json")
    key = "ctrl__ravdess__RR__r0"
    table = read_json(plan / "splits" / f"{key}.json")
    table["folds"][0]["test"] = table["folds"][0]["test"][::-1]
    rewrite(plan / "splits" / f"{key}.json", json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return {"integrity_fail": ["I2"]}


def case_speaker_swap(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="cnn", cell="GG", r=0, seed_index=0, fold=1)[0]
    lines = pred_lines(run, u["unit_id"])
    parts = lines[1].split(","); parts[2] = "99"
    lines[1] = ",".join(parts)
    write_pred(run, u["unit_id"], lines)
    return {"integrity_fail": ["I3"]}


def _unlink_done(run: Path, units: list[dict]) -> int:
    n = 0
    for u in units:
        d = run / "units" / u["unit_id"] / "DONE"
        if d.exists():
            d.unlink(); n += 1
    return n


def case_hpo_missing_config(plan, run, ctx):
    us = pick_units(ctx["plan"], arm="HPO", corpus_level="cremad", cell="GG_hpo", fold=0)
    if not us or _unlink_done(run, us[:1]) == 0:
        return {"skip": "HPO arm absent in this run"}
    return {"integrity_fail": ["I6"], "verdict": {"N14": "not tested"}}


def case_conditional_truncation(plan, run, ctx):
    for u in pick_units(ctx["plan"], arm="FT", corpus_level="ravdess", seed_index=1):
        d = run / "units" / u["unit_id"] / "DONE"
        if d.exists():
            d.unlink()
    return {"tested": ["N11", "R09"], "n": {"N11": 24}}


def case_hygiene_tamper(plan, run, ctx):
    src = ctx["manifests"] / "ravdess_manifest.csv"
    rows = read_csv(src)
    rows[0]["sha256"] = rows[1]["sha256"]      # two files, same bytes, different labels? force conflict
    if rows[0]["label"] == rows[1]["label"]:
        rows[0]["sha256"] = rows[2]["sha256"]
    import csv, io
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    mdir = run / "mutated_manifests"; mdir.mkdir(exist_ok=True)
    for f in ctx["manifests"].glob("*_manifest.csv"):
        os.link(f, mdir / f.name) if not (mdir / f.name).exists() else None
    (mdir / "ravdess_manifest.csv").unlink()
    atomic_write_text(mdir / "ravdess_manifest.csv", buf.getvalue())
    ctx["manifests_override"] = mdir
    return {"integrity_fail": ["I3"]}


def case_crossing_isolation(plan, run, ctx):
    for u in pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="cnn"):
        if u["seed_index"] != u["r"]:
            lines = pred_lines(run, u["unit_id"])
            new = [lines[0]]
            for ln in lines[1:]:
                parts = ln.split(","); logits = [float(x) for x in parts[5:]]
                worst = int(np.argmin(logits)); logits[worst] = max(logits) + 5.0
                parts[4] = str(worst); parts[5:] = [repr(v) for v in logits]
                new.append(",".join(parts))
            write_pred(run, u["unit_id"], new)
    return {"unchanged": ["R01", "R02", "R03"], "changed_descriptor": "D14"}


def case_logit_column_missing(plan, run, ctx):
    u = pick_units(ctx["plan"], arm="CTRL", corpus_level="ravdess", model="cnn", cell="RR", r=2, seed_index=2, fold=3)[0]
    lines = pred_lines(run, u["unit_id"])
    new = [",".join(ln.split(",")[:-1]) for ln in lines]
    write_pred(run, u["unit_id"], new)
    return {"integrity_fail": ["I4"]}


def case_family_placeholder(plan, run, ctx):
    for u in pick_units(ctx["plan"], arm="PROBECPU", corpus_level="ravdess"):
        d = run / "units" / u["unit_id"] / "DONE"
        if d.exists():
            d.unlink()
    return {"verdict": {"N15": "not tested"}, "family_size": {"N15": ctx["family_size"]("N15")}, "unchanged": ["N01"]}


def case_tost_missing(plan, run, ctx):
    us = pick_units(ctx["plan"], arm="PROBECPU", corpus_level="cremad", cell="G5")
    if not us or _unlink_done(run, us) == 0:
        return {"skip": "cremad PROBECPU absent in this run"}
    return {"verdict": {"T01": "not tested"}}


def case_mech_condition_missing(plan, run, ctx):
    us = pick_units(ctx["plan"], arm="MECH2X2", corpus_level="cremad", cell="spk")
    if not us or _unlink_done(run, us) == 0:
        return {"skip": "MECH2X2 absent in this run"}
    return {"verdict": {"N06": "not tested"}, "tested": ["N05"]}


CASES = [
    ("M01_drop_row", case_drop_row), ("M02_label_swap", case_label_swap), ("M03_fold_overlap", case_fold_overlap),
    ("M04_config_drift", case_config_drift), ("M05_duplicate_row", case_duplicate_row), ("M06_missing_cell", case_missing_cell),
    ("M07_done_tamper", case_done_tamper), ("M08_argmax_violation", case_argmax_violation),
    ("M09_split_table_tamper", case_split_table_tamper), ("M10_speaker_swap", case_speaker_swap),
    ("M11_hpo_missing_config", case_hpo_missing_config), ("M15_conditional_truncation", case_conditional_truncation),
    ("M20_hygiene_tamper", case_hygiene_tamper), ("M21_crossing_isolation", case_crossing_isolation),
    ("M24_logit_column_missing", case_logit_column_missing), ("M18_family_placeholder", case_family_placeholder),
    ("M23_tost_missing", case_tost_missing), ("M25_mech_condition_missing", case_mech_condition_missing),
]


# ----------------------------------------------------------------------------- world-level cases

def world_cases(plan: Path, manifests: Path, work: Path, registry: Path) -> list[dict]:
    out = []
    # M13 null world: no directional hypothesis may be supported on ravdess CTRL
    null_run = work / "world_null"
    if not (null_run / "run.json").exists():
        fixtures.generate(plan, manifests, null_run, fixtures.null_world(), {"CTRL"}, {"ravdess"}, run_id="null")
    res = score(plan, manifests, null_run, registry)
    supported = [h for h in ("R01", "R02", "R03", "R07") if res["hypotheses"][h]["verdict"] == "supported"]
    out.append({"case": "M13_null_world_no_false_support", "pass": len(supported) == 0, "detail": {"supported": supported,
                "means": {h: res["hypotheses"][h].get("mean") for h in ("R01", "R02", "R03", "R07")}}})
    # M14 sign flip: negative leakage must read 'not supported' with mean < 0
    w = fixtures.load_world(None)
    w["cell_shift"] = {k: (-v if k in ("RR", "RG", "RO", "P1RR", "RR_hpo") else v) for k, v in w["cell_shift"].items()}
    flip_run = work / "world_flip"
    if not (flip_run / "run.json").exists():
        fixtures.generate(plan, manifests, flip_run, w, {"CTRL"}, {"ravdess"}, run_id="flip")
    res = score(plan, manifests, flip_run, registry)
    r01 = res["hypotheses"]["R01"]
    out.append({"case": "M14_sign_flip_directional_reading", "pass": r01["verdict"] == "not supported" and r01["mean"] < 0,
                "detail": {"mean": r01["mean"], "reject": r01["reject"], "verdict": r01["verdict"]}})
    return out


# ----------------------------------------------------------------------------- driver

def evaluate(expect: dict, res: dict, base: dict) -> tuple[bool, dict]:
    detail, ok = {}, True
    for chk in expect.get("integrity_fail", []):
        got = not res["integrity"][chk]["pass"]
        detail[f"integrity_{chk}_failed"] = got
        ok &= got
    for hid, v in expect.get("verdict", {}).items():
        got = ({**res["hypotheses"], **res["tost"]})[hid]["verdict"]
        detail[f"verdict_{hid}"] = got
        ok &= got == v
    for hid, m in expect.get("family_size", {}).items():
        got = res["hypotheses"][hid]["m"]
        detail[f"m_{hid}"] = got
        ok &= got == m
    for hid in expect.get("unchanged", []):
        a, b = res["hypotheses"][hid], base["hypotheses"][hid]
        same = a.get("tested") == b.get("tested") and (not a.get("tested") or abs(a["mean"] - b["mean"]) < 1e-12)
        detail[f"unchanged_{hid}"] = same
        ok &= same
    for hid in expect.get("tested", []):
        got = ({**res["hypotheses"], **res["tost"]})[hid].get("tested", False)
        detail[f"tested_{hid}"] = got
        ok &= bool(got)
    for hid, n in expect.get("n", {}).items():
        got = res["hypotheses"][hid].get("n")
        detail[f"n_{hid}"] = got
        ok &= got == n
    if expect.get("changed_descriptor") == "D14":
        a, b = res["descriptors"]["D14"]["ravdess"], base["descriptors"]["D14"]["ravdess"]
        changed = a.get("tested") and b.get("tested") and a["M"] != b["M"]
        detail["D14_changed"] = bool(changed)
        ok &= bool(changed)
    return ok, detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--skip-world", action="store_true")
    ap.add_argument("--cases", default="", help="comma-separated case-name prefixes to run (default: all)")
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    base = score(a.plan, a.manifests, a.run, a.registry)
    plan_rows = read_csv(a.plan / "run_plan.csv")
    reg = {r["hypothesis_id"]: r for r in read_csv(a.registry / "hypothesis_registry.csv")}
    family_size = lambda hid: int(reg[hid]["family_size"])
    selected = [c for c in a.cases.split(",") if c]
    report = []
    for name, fn in CASES:
        if selected and not any(name.startswith(c) for c in selected):
            continue
        run_copy = a.work / name / "run"; plan_copy = a.work / name / "plan"
        if run_copy.exists():
            shutil.rmtree(run_copy.parent)
        link_tree(a.run, run_copy); link_tree(a.plan, plan_copy)
        ctx = {"plan": plan_rows, "manifests": a.manifests, "family_size": family_size}
        expect = fn(plan_copy, run_copy, ctx)
        if "skip" in expect:
            report.append({"case": name, "skipped": expect["skip"]})
            continue
        manifests = ctx.get("manifests_override", a.manifests)
        try:
            res = score(plan_copy, manifests, run_copy, a.registry)
            ok, detail = evaluate(expect, res, base)
        except Exception as exc:  # a crash is a failure to reject cleanly
            ok, detail = False, {"exception": repr(exc)[:300]}
        report.append({"case": name, "pass": ok, "expect": expect, "detail": detail})
        print(("PASS" if ok else "FAIL"), name, json.dumps(detail)[:200])
    if not a.skip_world:
        for item in world_cases(a.plan, a.manifests, a.work, a.registry):
            report.append(item)
            print(("PASS" if item["pass"] else "FAIL"), item["case"], json.dumps(item["detail"])[:200])
    n_pass = sum(1 for r in report if r.get("pass"))
    n_run = sum(1 for r in report if "pass" in r)
    summary = {"rejected_or_behaved": n_pass, "cases_run": n_run, "skipped": [r["case"] for r in report if "skipped" in r], "report": report}
    atomic_write_json(a.work / "mutation_report.json", summary)
    print(f"mutation battery: {n_pass}/{n_run} PASS, skipped {len(summary['skipped'])}")
    return 0 if n_pass == n_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
