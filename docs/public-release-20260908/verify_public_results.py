#!/usr/bin/env python3
"""Recompute the fixed Holm-6 and Holm-10 families from published draw tables.

Python >=3.10; install requirements-replay.txt, then run this file with --repo .
No project modules, prediction arrays, model weights, or network are used.
The tables already average five folds per draw: this script cannot verify that
upstream aggregation, checkpoint selection, or the authenticity of the archive.
Numerical agreement is not an independent replication of the experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

import scipy
from scipy.stats import t as student_t

SCOPE = "published draw tables only, not logits/retraining/archive authentication"
CORPORA = ("cremad", "subesco", "ravdess")
DRAWS = set(range(24))
OLD_DIR = Path("v3/final_program_20260907/reports/scores")
NEW_DIR = Path("docs/autodl-supplement-20260908/results")
INPUTS = (OLD_DIR / "draws.csv", OLD_DIR / "results.json",
          NEW_DIR / "draws.csv", NEW_DIR / "primary_tests.csv", NEW_DIR / "results.json")
OLD_COLUMNS = ("corpus", "draw", "delta_CE_pp", "delta_UAR_pp", "J_pp",
               "last_outer_uar", "middle_8_10_minus_late_14_15_pp")
METRICS = ("D_CE", "D_UAR", "J", "T_seen_ce", "T_unseen_ce",
           "T_seen_uar", "T_unseen_uar", "T_last")
NEW_COLUMNS = ("corpus", "comparison", "model", "window", "draw", "units",
               *(name + "_pp" for name in METRICS))
STAT_FIELDS = ("mean_pp", "sd_draw_pp", "se_pp", "t_statistic", "p_two_sided",
               "pointwise_95_ci_pp")
# Decimal CSV exports have lost the scorers' exact rational representation.
# This fixed tolerance covers floating summation, not rounded manuscript tables.
ATOL, RTOL = 1e-10, 1e-9


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def number(value, context):
    require(not isinstance(value, bool), f"{context}: boolean is not a number")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise VerificationError(f"{context}: invalid number {value!r}") from exc
    require(math.isfinite(result), f"{context}: nonfinite number")
    return result


def json_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(text):
    def reject_constant(value):
        raise VerificationError(f"nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=json_object, parse_constant=reject_constant)


def csv_rows(text, label, columns=None):
    reader = csv.DictReader(io.StringIO(text, newline=""))
    names = reader.fieldnames
    require(names and len(names) == len(set(names)), f"{label}: invalid/duplicate columns")
    if columns is not None:
        require(tuple(names) == tuple(columns), f"{label}: unexpected columns")
    rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            f"{label}: malformed CSV row")
    return rows


def agree(actual, expected, label):
    """Strict null/list handling; all compared numeric scalars must be finite."""
    if expected is None:
        require(actual is None, f"{label}: expected null, got {actual!r}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: list shape")
        for i, (left, right) in enumerate(zip(actual, expected)):
            agree(left, right, f"{label}[{i}]")
    else:
        left, right = number(actual, label), number(expected, label)
        require(math.isclose(left, right, abs_tol=ATOL, rel_tol=RTOL),
                f"{label}: published {left:.17g}, recomputed {right:.17g}")


def estimate(values):
    require(len(values) == 24, "each test must use exactly 24 draw effects")
    values = [number(v, "draw effect") for v in values]
    average = math.fsum(values) / 24
    sd = statistics.stdev(values)
    se = sd / math.sqrt(24)
    result = dict(mean_pp=average, sd_draw_pp=sd, se_pp=se, n_draws=24, df=23,
                  t_statistic=None, p_two_sided=None, pointwise_95_ci_pp=None)
    if sd != 0:
        statistic = average / se
        half = float(student_t.ppf(0.975, 23)) * se
        result.update(t_statistic=statistic,
                      p_two_sided=float(2 * student_t.sf(abs(statistic), 23)),
                      pointwise_95_ci_pp=[average - half, average + half])
    return result


def holm(estimates, family_size):
    require(len(estimates) == family_size, "wrong fixed Holm family size")
    # Undefined zero-variance tests keep their family slot as p=1 for ordering,
    # but retain null adjusted p and never reject (the prespecified convention).
    order = sorted(range(family_size), key=lambda i: (
        1.0 if estimates[i]["p_two_sided"] is None else estimates[i]["p_two_sided"], i))
    adjusted, running = [None] * family_size, 0.0
    for rank, index in enumerate(order):
        value = estimates[index]["p_two_sided"]
        running = min(1.0, max(running, (family_size - rank) * (1.0 if value is None else value)))
        if value is not None:
            adjusted[index] = running
    return adjusted


def keyed_tests(rows, key_function, expected, label):
    result = {}
    for row in rows:
        key = key_function(row)
        require(key not in result, f"{label}: duplicate test {key}")
        result[key] = row
    require(set(result) == set(expected), f"{label}: missing or unexpected test IDs")
    return result


def check_stats(row, recomputed, adjusted, holm_field, label, success_status):
    require(type(row.get("n_draws")) is int and row["n_draws"] == 24, f"{label}: n_draws")
    require(type(row.get("df")) is int and row["df"] == 23, f"{label}: df")
    for field in STAT_FIELDS:
        require(field in row, f"{label}: missing {field}")
        agree(row[field], recomputed[field], f"{label}.{field}")
    require(holm_field in row, f"{label}: missing adjusted p")
    agree(row[holm_field], adjusted, f"{label}.{holm_field}")
    reject = adjusted is not None and adjusted < 0.05
    require(type(row.get("reject_familywise_05")) is bool and row["reject_familywise_05"] == reject,
            f"{label}: rejection decision mismatch")
    status = success_status if recomputed["sd_draw_pp"] else "undefined_t_zero_sample_variance"
    require(row.get("status") == status, f"{label}: status mismatch")


def check_old(texts):
    rows = csv_rows(texts[OLD_DIR / "draws.csv"], "384 draws", OLD_COLUMNS)
    indexed = {}
    for row in rows:
        require(row["draw"] in {str(d) for d in DRAWS}, "384 draws: invalid draw ID")
        key = (row["corpus"], int(row["draw"]))
        require(key not in indexed, f"384 draws: duplicate {key}")
        indexed[key] = {k: number(row[k], f"384 {key}.{k}") for k in OLD_COLUMNS[2:]}
        agree(indexed[key]["J_pp"], indexed[key]["delta_CE_pp"] - indexed[key]["delta_UAR_pp"],
              f"384 {key}: J identity")
    require(set(indexed) == {(c, d) for c in CORPORA for d in DRAWS},
            "384 draws: expected exactly three corpora x draw IDs 0..23")
    doc = read_json(texts[OLD_DIR / "results.json"])
    require(doc.get("schema") == "ser-final-program-score-1", "384 result schema")
    keys = [(c, endpoint) for c in CORPORA for endpoint in ("delta_CE", "J")]
    published = keyed_tests(doc["tests"], lambda r: (r["corpus"], r["estimand"]), keys, "384 tests")
    estimates = [estimate([indexed[c, d][endpoint + "_pp"] for d in range(24)])
                 for c, endpoint in keys]
    adjusted = holm(estimates, 6)
    for key, value, adj in zip(keys, estimates, adjusted):
        check_stats(published[key], value, adj, "holm_p_six", f"384 {key}", "estimated")
    return [dict(id=f"{c}_{e}", **value, holm_p_six=adj)
            for (c, e), value, adj in zip(keys, estimates, adjusted)]


def check_new(texts):
    rows = csv_rows(texts[NEW_DIR / "draws.csv"], "720 draws", NEW_COLUMNS)
    groups = {(c, "model_window", model, "15") for c in CORPORA
              for model in ("wavlm_base_plus", "hubert_base")}
    groups |= {(c, "hubert_minus_wavlm15", "hubert_minus_wavlm", "15") for c in CORPORA}
    groups |= {(c, comparison, "wavlm_base_plus", window) for c in ("subesco", "ravdess")
               for comparison, window in (("model_window", "45"), ("window45_minus15", "45_minus15"))}
    indexed = {}
    for row in rows:
        group = tuple(row[k] for k in ("corpus", "comparison", "model", "window"))
        require(row["draw"] in {str(d) for d in DRAWS}, "720 draws: invalid draw ID")
        key = (*group, int(row["draw"]))
        require(key not in indexed, f"720 draws: duplicate {key}")
        indexed[key] = {m: number(row[m + "_pp"], f"720 {key}.{m}") for m in METRICS}
        expected_units = {m: "UAR_percent" if m.startswith("T_") and group[1] == "model_window"
                          else "percentage_points" for m in METRICS}
        require(read_json(row["units"]) == expected_units, f"720 {key}: units mismatch")
        values = indexed[key]
        agree(values["J"], values["D_CE"] - values["D_UAR"], f"720 {key}: J identity")
        agree(values["D_CE"], values["T_seen_ce"] - values["T_unseen_ce"], f"720 {key}: CE identity")
        agree(values["D_UAR"], values["T_seen_uar"] - values["T_unseen_uar"], f"720 {key}: UAR identity")
    require(set(indexed) == {(*group, d) for group in groups for d in DRAWS},
            "720 draws: expected exact 13 groups x draw IDs 0..23 (312 rows)")

    # Recreate paired differences by matching draw IDs, rather than trusting the
    # precomputed contrast rows. Check every published contrast metric as well.
    for c in CORPORA:
        for d in range(24):
            base = indexed[c, "model_window", "wavlm_base_plus", "15", d]
            comparisons = [("hubert_minus_wavlm15", "hubert_minus_wavlm", "15", "hubert_base", "15")]
            if c != "cremad":
                comparisons.append(("window45_minus15", "wavlm_base_plus", "45_minus15", "wavlm_base_plus", "45"))
            for kind, model, window, left_model, left_window in comparisons:
                left = indexed[c, "model_window", left_model, left_window, d]
                difference = indexed[c, kind, model, window, d]
                for metric in METRICS:
                    agree(difference[metric], left[metric] - base[metric], f"720 {c}/{d}/{kind}/{metric}")

    specs = []
    for c in CORPORA:
        for endpoint in ("D_CE", "J"):
            specs.append((f"hubert_{c}_{endpoint}", c, endpoint, "hubert_base", 15,
                          "absolute_seen_minus_unseen", (c, "model_window", "hubert_base", "15"), endpoint))
    for c in ("subesco", "ravdess"):
        for endpoint, metric in (("L_CE", "D_CE"), ("L_J", "J")):
            specs.append((f"window_{c}_{endpoint}", c, endpoint, "wavlm_base_plus", None,
                          "paired_window45_minus_window15", (c, "window45_minus15", "wavlm_base_plus", "45_minus15"), metric))
    estimates = []
    for _, c, _, model, window, _, group, metric in specs:
        if window is None:
            values = [indexed[c, "model_window", model, "45", d][metric]
                      - indexed[c, "model_window", model, "15", d][metric] for d in range(24)]
        else:
            values = [indexed[(*group, d)][metric] for d in range(24)]
        estimates.append(estimate(values))
    adjusted = holm(estimates, 10)
    doc = read_json(texts[NEW_DIR / "results.json"])
    require(doc.get("schema") == "ser-autodl-supplement-results-1", "720 result schema")
    ids = [s[0] for s in specs]
    tables = [("results.json", keyed_tests(doc["tests"], lambda r: r["id"], ids, "720 JSON"))]
    csv_tests = csv_rows(texts[NEW_DIR / "primary_tests.csv"], "720 primary CSV")
    converted = []
    for row in csv_tests:
        converted_row = dict(row)
        for field in (*STAT_FIELDS, "holm_p_ten", "n_draws", "df", "window"):
            require(field in row, f"720 primary CSV: missing {field}")
            converted_row[field] = None if row[field] == "NA" else read_json(row[field])
        require(row.get("reject_familywise_05") in ("True", "False"), "720 CSV: invalid boolean")
        converted_row["reject_familywise_05"] = row["reject_familywise_05"] == "True"
        converted.append(converted_row)
    tables.append(("primary_tests.csv", keyed_tests(converted, lambda r: r["id"], ids, "720 CSV")))
    for source, table in tables:
        for spec, value, adj in zip(specs, estimates, adjusted):
            uid, c, endpoint, model, window, comparison, _, _ = spec
            row = table[uid]
            for field, expected in (("corpus", c), ("endpoint", endpoint), ("model", model),
                                    ("window", window), ("comparison", comparison), ("unit", "percentage_points")):
                require(row.get(field) == expected, f"720 {source}/{uid}: wrong {field}")
            check_stats(row, value, adj, "holm_p_ten", f"720 {source}/{uid}", "inferential_estimate")
    return [dict(id=spec[0], **value, holm_p_ten=adj)
            for spec, value, adj in zip(specs, estimates, adjusted)]


def verify(repo):
    repo = Path(repo).resolve()
    payloads = {relative: (repo / relative).read_bytes() for relative in INPUTS}
    texts = {relative: data.decode("utf-8-sig") for relative, data in payloads.items()}
    six, ten = check_old(texts), check_new(texts)
    require(all((repo / name).read_bytes() == data for name, data in payloads.items()),
            "an input changed during verification")
    return dict(pass_numeric_replay=True, scope=SCOPE,
                inference="24 published draw means; pointwise t intervals; separate fixed Holm-6 and Holm-10 families",
                tolerances=dict(absolute=ATOL, relative=RTOL), python=sys.version.split()[0], scipy=scipy.__version__,
                input_sha256={str(p).replace("\\", "/"): hashlib.sha256(b).hexdigest() for p, b in payloads.items()},
                draw_rows={"study384": 72, "study720": 312},
                family6=six, family10=ten)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2],
                        help="repository root (default: inferred from this script's location)")
    parser.add_argument("--json", action="store_true", help="print all recomputed statistics and input hashes")
    args = parser.parse_args()
    try:
        result = verify(args.repo)
    except (VerificationError, OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"FAIL: {exc}\nScope: {SCOPE}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print("PASS: 6 + 10 prespecified primary tests match the published results.")
        print("Checked: exact 24-draw groups, mean/SD/SE/t/two-sided p/pointwise 95% CI, separate Holm-6/Holm-10.")
        print("Draw rows: study384=72; study720=312. No pilot or additional hypothesis tests.")
        print(f"Scope: {SCOPE}")
        print(f"Python {result['python']}; SciPy {result['scipy']}; atol={ATOL:g}, rtol={RTOL:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
