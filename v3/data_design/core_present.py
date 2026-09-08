"""Immutable paper tables/figures from a complete independently approved Study II."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
import ctypes
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.data_design import core_score, core_verify

require, read_json, sha = core_verify.require, core_verify.read_json, core_verify.byte_hash
MODELS = ("ridge_wavlm", "cnn")
MODEL_NAMES = {"ridge_wavlm": "Ridge–WavLM", "cnn": "CNN"}
POLICIES = tuple(itertools.product(MODELS, (288, 576), (12, 48), ("prompt_seen", "prompt_new")))
CONTRASTS = tuple((m, b, c, s) for m in MODELS for b in (288, 576)
                  for c, s in (("S48_minus_S12", "prompt_new"), ("S48_minus_S12", "prompt_seen"),
                               ("new_minus_seen_of_S48_minus_S12", "")))
ARTIFACTS = ("results.csv", "draw_estimates.csv", "tables.md", "absolute_uar.png", "absolute_uar.pdf",
             "allocation_contrasts.png", "allocation_contrasts.pdf", "presentation_identity.json")
NOTE = "Pointwise 95% conditional bootstrap intervals; n = 91 speakers. Draws are not independent samples."


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def constrain_process():
    torch = sys.modules.get("torch")
    require(torch is None or not torch.cuda.is_initialized(), "CUDA already initialized")
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        require(bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)), "cannot set BelowNormal priority")


def checked_snapshot(snapshot):
    for name, expected in snapshot.items():
        path = Path(name)
        require(path.is_file() and sha(path) == expected, f"verified artifact changed: {path.name}")


def parse_score_payload(payload):
    """Interpret only the exact bytes already checked against both approval receipts."""
    return json.loads(payload.decode("utf-8-sig"), object_pairs_hook=core_verify._object,
                      parse_constant=lambda value: require(False, f"nonfinite JSON: {value}"))


def required_snapshot_paths(repo, plan_path, run, analysis, plan):
    paths = {plan_path, repo / "v3/data_design/result_verify.py", repo / "v3/data_design/core_score.py"}
    paths.update(repo / p for p in plan["source_sha256"])
    paths.update(repo / plan["input"][k] for k in ("manifest_path", "demographics_path"))
    paths.update(run / p for p in ("completion.json", "analysis_lock.json", "run_identity.json", "ledger.jsonl"))
    paths.update(run / f"environment_{m}.json" for m in MODELS)
    paths.update(analysis / p for p in ("score.json", "SCORE_DONE", "score_identity.json", "per_speaker.csv"))
    paths.update(run / "units" / u["unit_id"] / p for u in plan["units"]
                 for p in ("DONE", "unit.json", "predictions.npz"))
    return {str(p.resolve()) for p in paths}


def load_approved(repo, plan_path, run, analysis):
    """All closure/approval/binding checks precede parsing scientific score.json."""
    repo, plan_path, run, analysis = (Path(p).resolve() for p in (repo, plan_path, run, analysis))
    metadata = core_verify.verify_plan_file(repo, plan_path, metadata_only=True)
    plan = read_json(plan_path)
    require(plan.get("design", {}).get("runtime_complete") is True
            and plan["plan_sha256"] == metadata["plan_sha256"]
            == hashlib.sha256(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})).hexdigest(),
            "unlocked or changed plan")
    closure, _ = core_score.verify_closed(plan, run)  # Byte hashes only, no NPZ interpretation.
    approval_path = analysis / "verification.json"
    require(approval_path.is_file(), "independent verification is required before presentation")
    approval_sha = sha(approval_path)
    approval = read_json(approval_path)
    require(approval.get("schema") == "ser-study2-independent-result-verification-1"
            and approval.get("pass") is True and approval.get("results_approved_for_paper") is True,
            "independent results are not approved for paper")
    require(approval.get("scientific_arrays_read") is True and type(approval.get("numeric_fields_checked")) is int
            and approval["numeric_fields_checked"] > 0 and approval.get("tolerance_pp") == 1e-10
            and type(approval.get("max_numeric_error")) in (int, float)
            and 0 <= approval["max_numeric_error"] <= 1e-10, "independent numeric replay is incomplete")
    require(all(approval.get(k) == n for k, n in {
        "n_units": 1440, "n_speakers": 91, "absolute_estimates": 16,
        "speaker_count_contrasts": 8, "interaction_contrasts": 4, "per_speaker_rows": 1456}.items()),
        "independent verification population mismatch")
    require(approval.get("verifier_sha256") == sha(repo / "v3/data_design/result_verify.py"),
            "independent verifier version changed")
    snapshot = approval.get("file_hashes", {})
    require(isinstance(snapshot, dict) and set(snapshot) == required_snapshot_paths(repo, plan_path, run, analysis, plan),
            "independent verification snapshot is incomplete or belongs to other paths")
    checked_snapshot(snapshot)
    identity = read_json(analysis / "score_identity.json")
    require(approval.get("identity") == identity and all(identity.get(k) == v for k, v in closure.items()),
            "analysis identity differs from approved complete closure")
    require(identity.get("scorer_sha256") == sha(repo / "v3/data_design/core_score.py"), "scorer version changed")
    source = {name: sha(repo / "v3/data_design" / name)
              for name in ("core_present.py", "core_score.py", "core_verify.py", "result_verify.py")}
    require(source["core_present.py"] == sha(__file__) and source["core_score.py"] == sha(core_score.__file__)
            and source["core_verify.py"] == sha(core_verify.__file__), "executing modules differ from recorded source identity")
    score_path = analysis / "score.json"
    score_payload = score_path.read_bytes()
    score_sha = hashlib.sha256(score_payload).hexdigest()
    require(score_sha == snapshot[str(score_path)], "scientific payload changed after approval snapshot check")
    csv_sha = sha(analysis / "per_speaker.csv")
    require(csv_sha == snapshot[str(analysis / "per_speaker.csv")], "approved per-speaker artifact changed during gating")
    require(read_json(analysis / "SCORE_DONE") == {
        "identity": identity, "score_sha256": score_sha, "per_speaker_sha256": csv_sha}, "SCORE_DONE binding mismatch")
    require(sha(approval_path) == approval_sha, "approval changed during gating")
    # This is the first interpretation of scientific values; do not reopen the path.
    report = parse_score_payload(score_payload)
    require(report.get("schema") == "ser-study2-score-1" and report.get("identity") == identity
            and report.get("per_speaker_sha256") == csv_sha, "scientific report identity mismatch")
    snapshot = dict(snapshot)
    snapshot[str(approval_path)] = approval_sha
    presentation_identity = {"schema": "ser-study2-presentation-identity-1", "score_identity": identity,
        "score_sha256": score_sha, "csv_sha256": csv_sha, "verification_sha256": approval_sha,
        "verification_snapshot_sha256": hashlib.sha256(canonical(approval["file_hashes"])).hexdigest(),
        "source_sha256": source, "ordering": "model,budget,S,scenario; contrasts new,seen,DID",
        "statistics_recomputed": False}
    return report, presentation_identity, snapshot


def ordered_rows(report):
    absolute, contrasts = {}, {}
    for row in report["absolute_uar"]:
        key = tuple(row[k] for k in ("model", "B", "S", "scenario"))
        require(key not in absolute, "duplicate absolute estimate")
        absolute[key] = row
    for row in report["contrasts"]:
        key = (row["model"], row["B"], row["contrast"], row.get("scenario", ""))
        require(key not in contrasts, "duplicate contrast estimate")
        contrasts[key] = row
    require(set(absolute) == set(POLICIES) and set(contrasts) == set(CONTRASTS), "all 16 absolute and 12 contrasts required")
    rows = [("absolute", absolute[k]) for k in POLICIES] + [("contrast", contrasts[k]) for k in CONTRASTS]
    for kind, row in rows:
        numbers = [row["estimate_pp"], *row["ci95_percentile_pp"], *row["draw_estimates_pp"], *row["draw_range_pp"]]
        require(len(numbers) == 8 and all(type(v) in (int, float) and math.isfinite(v) for v in numbers)
                and len(row["ci95_percentile_pp"]) == 2 and len(row["draw_estimates_pp"]) == 3
                and len(row["draw_range_pp"]) == 2 and row["n_speakers"] == 91, "malformed verified estimate")
        require(row["ci95_percentile_pp"][0] <= row["ci95_percentile_pp"][1], "reversed confidence interval")
        if kind == "contrast":
            primary = row["contrast"] == "S48_minus_S12" and row.get("scenario") == "prompt_new"
            require(row.get("primary") is primary, "primary contrast label mismatch")
    return rows


def label(row):
    if "S" in row:
        return f"S{row['S']} / {'seen' if row['scenario'] == 'prompt_seen' else 'new'}"
    if row["contrast"] == "new_minus_seen_of_S48_minus_S12":
        return "(new S48−S12) − (seen S48−S12)"
    return f"{'new (primary)' if row['scenario'] == 'prompt_new' else 'seen'}: S48−S12"


def write_tables(report, out):
    rows = ordered_rows(report)
    main, draws = io.StringIO(newline=""), io.StringIO(newline="")
    writer, supplemental = csv.writer(main, lineterminator="\n"), csv.writer(draws, lineterminator="\n")
    writer.writerow(["kind", "model", "B", "S", "scenario", "contrast", "primary", "unit", "estimate",
                     "ci95_low", "ci95_high", "n_speakers"])
    supplemental.writerow(["kind", "model", "B", "S", "scenario", "contrast", "unit",
                           "draw0", "draw1", "draw2", "draw_min", "draw_max", "n_speakers"])
    text = ["# Study II: complete verified estimates", "", NOTE, "",
            "All planned model/budget/condition combinations are retained. No p-values or simultaneous-coverage claims.",
            "CSV files preserve the supplied numerical precision; Markdown displays two decimal places.", "",
            "| Kind | Model | B | Allocation / contrast | Unit | Estimate [95% interval] |",
            "|---|---|---:|---|---|---|"]
    draw_text = ["", "## Complete-draw sensitivity", "", "These three draws reuse the same 91 speakers.", "",
                 "| Kind | Model | B | Allocation / contrast | Unit | Draw 0 | Draw 1 | Draw 2 | Range |",
                 "|---|---|---:|---|---|---:|---:|---:|---|"]
    for kind, row in rows:
        unit = "UAR %" if kind == "absolute" else "percentage points"
        fields = [kind, row["model"], row["B"], row.get("S", ""), row.get("scenario", ""), row.get("contrast", "")]
        writer.writerow([*fields, row.get("primary", False), unit, row["estimate_pp"],
                         *row["ci95_percentile_pp"], row["n_speakers"]])
        supplemental.writerow([*fields, unit, *row["draw_estimates_pp"], *row["draw_range_pp"], row["n_speakers"]])
        lo, hi = row["ci95_percentile_pp"]
        prefix = f"| {kind} | {MODEL_NAMES[row['model']]} | {row['B']} | {label(row)} | {unit} |"
        text.append(prefix + f" {row['estimate_pp']:.2f} [{lo:.2f}, {hi:.2f}] |")
        d0, d1, d2 = row["draw_estimates_pp"]
        low, high = row["draw_range_pp"]
        draw_text.append(prefix + f" {d0:.2f} | {d1:.2f} | {d2:.2f} | [{low:.2f}, {high:.2f}] |")
    (out / "results.csv").write_text(main.getvalue(), encoding="utf-8", newline="")
    (out / "draw_estimates.csv").write_text(draws.getvalue(), encoding="utf-8", newline="")
    (out / "tables.md").write_text("\n".join(text + draw_text) + "\n", encoding="utf-8", newline="\n")


def render_figures(report, out, *, test_fixture=False):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    ordered_rows(report)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
    lookup = {(r["model"], r["B"], r["S"], r["scenario"]): r for r in report["absolute_uar"]}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, (model, budget) in zip(axes.flat, itertools.product(MODELS, (288, 576))):
        for scenario, color, marker, offset in (("prompt_seen", "#2166ac", "o", -.045),
                                                ("prompt_new", "#b35806", "s", .045)):
            points = [lookup[(model, budget, s, scenario)] for s in (12, 48)]
            xs = [offset, 1 + offset]
            ax.plot(xs, [r["estimate_pp"] for r in points], marker=marker, color=color,
                    linestyle="-" if scenario == "prompt_seen" else "--", label="Seen sentences" if scenario == "prompt_seen" else "Unseen sentences")
            for x, r in zip(xs, points):
                lo, hi = r["ci95_percentile_pp"]
                ax.vlines(x, lo, hi, color=color, linewidth=1.6)
                ax.hlines([lo, hi], x - .03, x + .03, color=color, linewidth=1.6)
        ax.set(title=f"{MODEL_NAMES[model]} · B = {budget}", xticks=[0, 1], xticklabels=["12", "48"],
               xlabel="Training speakers", ylabel="Speaker-weighted UAR (%)", ylim=(0, 100), xlim=(-.22, 1.22))
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Fixed-budget speaker allocation: all absolute UAR estimates", y=.985, fontsize=14)
    fig.text(.5, .02, NOTE, ha="center", fontsize=9)
    if test_fixture:
        fig.text(.5, .51, "TEST FIXTURE — NOT RESEARCH RESULTS", ha="center", color="#a50000", fontsize=15, alpha=.7)
    fig.tight_layout(rect=(0, .05, 1, .95))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"absolute_uar.{extension}", dpi=220, metadata={"Creator": "SER Study II presentation", "CreationDate": None} if extension == "pdf" else None)
    plt.close(fig)
    lookup = {(r["model"], r["B"], r["contrast"], r.get("scenario", "")): r for r in report["contrasts"]}
    definitions = [("S48_minus_S12", "prompt_new", "Unseen sentences\nS48 − S12 (primary)"),
                   ("S48_minus_S12", "prompt_seen", "Seen sentences\nS48 − S12"),
                   ("new_minus_seen_of_S48_minus_S12", "", "Difference in differences\nΔnew − Δseen")]
    row_keys = list(itertools.product(MODELS, (288, 576)))
    span = max(1., max(abs(v) for r in report["contrasts"] for v in [r["estimate_pp"], *r["ci95_percentile_pp"]])) * 1.16
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, sharex=True)
    for ax, (contrast, scenario, title) in zip(axes, definitions):
        ax.axvline(0, color="#777777", linewidth=.9, linestyle="--")
        for y, (model, budget) in enumerate(row_keys):
            row = lookup[(model, budget, contrast, scenario)]
            color = "#2166ac" if model == "ridge_wavlm" else "#b35806"
            lo, hi = row["ci95_percentile_pp"]
            ax.hlines(y, lo, hi, color=color, linewidth=2)
            ax.vlines([lo, hi], y - .08, y + .08, color=color)
            ax.plot(row["estimate_pp"], y, "o" if model == "ridge_wavlm" else "s", color=color)
        ax.set_title(title)
        ax.set_xlabel("UAR difference (percentage points)")
        ax.set_xlim(-span, span)
        ax.set_yticks(range(4), [f"{MODEL_NAMES[m]} · B={b}" for m, b in row_keys])
        ax.grid(axis="x", color="#eeeeee", linewidth=.6)
    axes[0].invert_yaxis()
    fig.suptitle("All specified speaker-allocation contrasts", y=.995, fontsize=14)
    fig.text(.5, .035, NOTE, ha="center", fontsize=9)
    if test_fixture:
        fig.text(.5, .49, "TEST FIXTURE — NOT RESEARCH RESULTS", ha="center", color="#a50000", fontsize=15, alpha=.7)
    fig.tight_layout(rect=(0, .09, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"allocation_contrasts.{extension}", dpi=220, metadata={"Creator": "SER Study II presentation", "CreationDate": None} if extension == "pdf" else None)
    plt.close(fig)
    return matplotlib.__version__


def verify_existing(out, identity):
    require(out.is_dir() and {p.name for p in out.iterdir()} == set(ARTIFACTS) | {"PRESENTATION_DONE.json"},
            "existing presentation is incomplete or has extra files; refusing overwrite")
    done = read_json(out / "PRESENTATION_DONE.json")
    require(done.get("schema") == "ser-study2-presentation-done-1" and done.get("identity") == identity
            and read_json(out / "presentation_identity.json") == identity, "existing presentation has different identity")
    hashes = done.get("artifact_sha256", {})
    require(set(hashes) == set(ARTIFACTS), "presentation artifact inventory mismatch")
    for name, expected in hashes.items():
        require((out / name).is_file() and sha(out / name) == expected, "existing presentation artifact is corrupt")
    return done


def present(repo, plan_path, run, analysis, out):
    constrain_process()
    repo, plan_path, run, analysis, out = (Path(p).resolve() for p in (repo, plan_path, run, analysis, out))
    expected = repo / "v3/data_design/work/presentation"
    require(expected.resolve() == expected and out == expected and not out.is_relative_to(run)
            and not out.is_relative_to(analysis) and not run.is_relative_to(out) and not analysis.is_relative_to(out),
            "output must be the dedicated new repo/v3/data_design/work/presentation directory")
    report, identity, snapshot = load_approved(repo, plan_path, run, analysis)
    import matplotlib
    from threadpoolctl import threadpool_limits
    identity["rendering_runtime"] = {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__, "threads": 2}
    if out.exists():
        return verify_existing(out, identity)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / (".presentation.tmp-" + uuid.uuid4().hex)
    stage.mkdir()
    with threadpool_limits(limits=2):
        write_tables(report, stage)
        render_figures(report, stage)
    (stage / "presentation_identity.json").write_bytes(canonical(identity) + b"\n")
    done = {"schema": "ser-study2-presentation-done-1", "identity": identity,
            "artifact_sha256": {name: sha(stage / name) for name in ARTIFACTS}}
    checked_snapshot(snapshot)
    require(not (run / ".core-run.lock").exists(), "runner restarted during rendering")
    for name, expected_sha in identity["source_sha256"].items():
        require(sha(repo / "v3/data_design" / name) == expected_sha, "source changed during rendering")
    (stage / "PRESENTATION_DONE.json").write_bytes(canonical(done) + b"\n")
    require(not out.exists(), "presentation output appeared during rendering")
    os.replace(stage, out)
    return verify_existing(out, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "run", "analysis", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    a = parser.parse_args(argv)
    done = present(a.repo, a.plan, a.run, a.analysis, a.out)
    print(json.dumps({"created_or_verified": True, "artifacts": len(done["artifact_sha256"]),
                      "plan_sha256": done["identity"]["score_identity"]["plan_sha256"]}))


if __name__ == "__main__":
    main()
