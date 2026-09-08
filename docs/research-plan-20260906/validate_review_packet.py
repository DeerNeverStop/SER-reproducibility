"""Read-only, standard-library integrity/arithmetic checks for this review packet.

Default: require manifest.json with files=[{path, sha256, optional size_bytes}].
Paths are canonical POSIX paths relative to the packet. List every regular file
including this validator, excluding manifest.json itself. Symlinks/junctions are
not accepted. --preflight skips the not-yet-created manifest and is NOT a seal.

No imports/execution of budget_plan.py, network calls, training or output files.
Markdown checks cover inline, reference and HTML href/src destinations outside
code. Fragments are not checked against generated heading IDs; remote links are
not fetched. These checks do not certify science, power, or new experiment runs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import NormalDist
from urllib.parse import unquote, urlsplit


class CheckError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(CheckError(f"nonfinite JSON: {value}")))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def in_packet(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def packet_files(root):
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            item = Path(directory) / name
            require(not item.is_symlink() and not getattr(item, "is_junction", lambda: False)(),
                    f"symlink/junction forbidden: {item.relative_to(root).as_posix()}")
        for name in files:
            item = Path(directory) / name
            require(item.is_file(), "non-regular packet entry")
            paths.append(item)
    return sorted(paths)


def check_manifest(root, files):
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), "manifest.json missing; use --preflight only before sealing")
    entries = read_json(manifest_path).get("files")
    require(isinstance(entries, list), "manifest.files must be a list")
    expected = {}
    for entry in entries:
        require(isinstance(entry, dict), "manifest file entry must be an object")
        name = entry.get("path")
        require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
                "manifest path must be a nonempty relative POSIX path")
        parts = PurePosixPath(name)
        require(not parts.is_absolute() and ".." not in parts.parts and parts.as_posix() == name,
                f"noncanonical or escaping manifest path: {name}")
        require(name != "manifest.json", "manifest must not include its own recursive hash")
        require(name not in expected, f"duplicate manifest path: {name}")
        require(re.fullmatch(r"[0-9a-fA-F]{64}", str(entry.get("sha256", ""))) is not None,
                f"invalid SHA256: {name}")
        require(in_packet(root / name, root), f"manifest target escapes packet: {name}")
        expected[name] = entry
    actual = {path.relative_to(root).as_posix(): path for path in files
              if path != manifest_path}
    require(set(expected) == set(actual),
            f"manifest inventory mismatch; missing={sorted(set(expected)-set(actual))}; "
            f"unlisted={sorted(set(actual)-set(expected))}")
    for name, entry in expected.items():
        require(sha256(actual[name]) == entry["sha256"].lower(), f"SHA256 mismatch: {name}")
        if "size_bytes" in entry:
            require(type(entry["size_bytes"]) is int and entry["size_bytes"] == actual[name].stat().st_size,
                    f"size mismatch: {name}")
    return {"files_checked": len(expected), "manifest_self_hash_excluded": True}


def without_code(text):
    kept, fence = [], None
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            kept.append("")
        elif marker:
            fence = marker[1]
            kept.append("")
        elif line.startswith("    ") or line.startswith("\t"):
            kept.append("")
        else:
            kept.append(line)
    return re.sub(r"(`+)(.*?)\1", "", "\n".join(kept), flags=re.S)


def balanced_end(text, start, opener, closer):
    depth, index = 1, start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == opener:
            depth += 1
        elif text[index] == closer:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def destination(body):
    body = body.lstrip()
    if body.startswith("<"):
        end = body.find(">")
        require(end >= 0, "unterminated angle-bracket link destination")
        raw = body[1:end]
    else:
        chars, depth, index = [], 0, 0
        while index < len(body):
            char = body[index]
            if char == "\\" and index + 1 < len(body):
                chars.append(body[index:index + 2]); index += 2; continue
            if char.isspace() and depth == 0:
                break
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            chars.append(char); index += 1
        raw = "".join(chars)
    return re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~ ])", r"\1", raw)


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls.extend(value for key, value in attrs if key.lower() in {"href", "src"} and value)


def markdown_destinations(text):
    text = without_code(text)
    refs, output, lines = {}, [], []
    normalize = lambda value: " ".join(value.split()).casefold()
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}\[([^\]]+)\]:\s*(.+)$", line)
        if match:
            key = normalize(match[1]); require(key not in refs, f"duplicate link reference: {key}")
            refs[key] = destination(match[2]); output.append(refs[key])
            lines.append("")
        else:
            lines.append(line)
    text = "\n".join(lines)
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 2; continue
        if text[index] != "[":
            index += 1; continue
        end = balanced_end(text, index, "[", "]")
        if end is None:
            index += 1; continue
        label, after = text[index + 1:end], end + 1
        if after < len(text) and text[after] == "(":
            finish = balanced_end(text, after, "(", ")")
            require(finish is not None, "unterminated Markdown link")
            output.append(destination(text[after + 1:finish])); index = finish + 1
        elif after < len(text) and text[after] == "[":
            finish = balanced_end(text, after, "[", "]")
            require(finish is not None, "unterminated reference link")
            key = normalize(text[after + 1:finish] or label)
            require(key in refs, f"undefined link reference: {key}")
            output.append(refs[key]); index = finish + 1
        else:
            if normalize(label) in refs:
                output.append(refs[normalize(label)])
            index = end + 1
    html = HTMLLinks(); html.feed(text); output.extend(html.urls)
    return output


def check_destination(raw, document, root, allow_missing_manifest=False):
    require("\x00" not in raw, "NUL in link")
    require(not re.match(r"^[A-Za-z]:[\\/]", raw), "local drive link forbidden")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() in {"http", "https", "mailto", "tel", "data"}:
        return False
    require(not parsed.scheme, f"unsupported/local URI scheme: {parsed.scheme}")
    require(not parsed.netloc, "network-path link forbidden")
    name = unquote(parsed.path).replace("\\", "/")
    require(not name.startswith("/") and not re.match(r"^[A-Za-z]:", name),
            "absolute local link forbidden")
    target = document.parent / name if name else document
    require(in_packet(target, root), f"local link escapes packet: {raw}")
    if allow_missing_manifest and target.resolve() == (root / "manifest.json").resolve() and not target.exists():
        return None
    require(target.exists(), f"missing local link: {raw}")
    return True


def check_links(root, files, allow_missing_manifest=False):
    local, remote, documents, deferred = 0, 0, 0, 0
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        documents += 1
        try:
            for url in markdown_destinations(path.read_text(encoding="utf-8-sig")):
                result = check_destination(url, path, root, allow_missing_manifest)
                if result is None: deferred += 1
                elif result: local += 1
                else: remote += 1
        except (CheckError, ValueError) as exc:
            raise CheckError(f"{path.relative_to(root).as_posix()}: {exc}") from exc
    return {"Markdown_files": documents, "local_destinations": local,
            "remote_destinations_not_fetched": remote, "fragment_IDs_checked": False,
            "manifest_links_deferred_in_preflight":deferred}


def close(actual, expected, label, abs_tol=1e-8):
    if isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected), f"length mismatch: {label}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            close(left, right, f"{label}[{index}]", abs_tol)
        return
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=abs_tol),
            f"arithmetic mismatch: {label}: got {actual}, expected {expected}")


def table_rows(text):
    return [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in text.splitlines() if line.strip().startswith("|")]


def displayed_numbers(cell):
    return [float(value) for value in re.findall(r"\d+(?:\.\d+)?", cell.replace(",", ""))]


def check_budget(root):
    b = read_json(root / "budget_plan.json"); p = b["parameters"]
    tree = ast.parse((root / "budget_plan.py").read_text(encoding="utf-8-sig"))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "P" for target in node.targets)]
    require(len(assignments) == 1 and ast.literal_eval(assignments[0].value) == p,
            "script literal P differs from budget_plan.json; script was not executed")
    m, rate = p["retry_slowdown_multiplier"], p["hourly_rate_USD"]
    require(m >= 1 and rate > 0, "invalid multiplier or hourly rate")
    a, e = p["E2"], b["E2"]
    for key, expected in {"pilot_CNN":3*1*1*3, "pilot_Ridge":3*1*1*3,
                          "core_CNN":3*5*6*3, "core_Ridge":3*5*6*3,
                          "optional_FT":3*5*6*1*2}.items():
        close(a[key], expected, f"E2 plan count {key}")
    close(e["core_fits"], a["core_CNN"]+a["core_Ridge"], "E2 core fits")
    close(e["with_FT_fits"], e["core_fits"]+a["optional_FT"], "E2 full fits")
    core, full = [], []
    for index in range(2):
        cnn = a["core_CNN"]*a["CNN_seconds_scenario"][index]/3600
        ft = a["optional_FT"]*a["FT_seconds_scenario"][index]/3600
        core.append((cnn+a["feature_GPU_hours"][index])*m+a["setup_transfer_GPU_hours"][index])
        full.append(core[-1]+ft*m)
    close(e["core_GPU_hours"], core, "E2 core hours")
    close(e["with_FT_GPU_hours"], full, "E2 full hours")
    close(e["with_FT_GPU_USD"], [value*rate for value in full], "E2 USD")
    initial = p["E0_E1_E2pilot_allocated_GPU_hours"]
    for key, values in [("E0_E1_pilot_plus_core_GPU_hours", core),
                        ("E0_E1_pilot_plus_full_GPU_hours", full)]:
        close(e[key], [value+initial[i] for i, value in enumerate(values)], f"E2 {key}")
    a, e = p["E3"], b["E3"]
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in a["generation_RTF_scenarios"]), "RTF scenarios must be finite and positive")
    panels = a["folds"]*a["draws"]
    for model in ["CNN", "Ridge"]:
        close(a[model], a["arms"]*panels, f"E3 {model} fits")
        close(a[f"pilot_{model}"], a["arms"]*1*3, f"E3 pilot {model}")
    close(a["optional_FT"], a["arms"]*a["folds"]*1*2, "E3 FT fits")
    close(e["core_fits"], a["CNN"]+a["Ridge"], "E3 core fits")
    close(e["with_optional_FT_fits"], e["core_fits"]+a["optional_FT"], "E3 full fits")
    clips = a["synthetic_arms"]*panels*a["accepted_clips_per_arm_split"]
    accepted = clips*a["seconds_per_clip_assumed"]/3600
    anchor = a["one_time_anchor_candidates"]*a["anchor_emotions"]*a["anchor_seconds_assumed"]/3600
    require(0 < a["acceptance_probability_assumed"] <= 1, "invalid acceptance rate")
    attempted = (accepted+anchor)/a["acceptance_probability_assumed"]
    qa = attempted*a["QA_RTF_all_attempts_assumed"]
    conventional = panels*a["accepted_clips_per_arm_split"]*a["seconds_per_clip_assumed"]/3600
    close(a["conventional_aug_audio_hours_assumed"], conventional, "conventional audio hours")
    features = (accepted+conventional)*a["feature_RTF_accepted_training_audio_assumed"]
    training = (a["CNN"]*a["CNN_seconds_assumed"]+a["optional_FT"]*a["FT_seconds_assumed"])/3600
    for key, expected in [("accepted_unique_clips_no_cross_split_reuse", clips),
                          ("accepted_audio_hours", accepted), ("one_time_accepted_anchor_audio_hours", anchor),
                          ("expected_attempted_audio_hours", attempted),
                          ("pilot_synthetic_clips", a["synthetic_arms"]*a["pilot_accepted_clips_per_synthetic_arm"]*a["pilot_synthesis_splits"])]:
        close(e[key], expected, f"E3 {key}")
    close(a["pilot_rating_clips"], e["pilot_synthetic_clips"]+120, "pilot rating composition")
    close(a["formal_rating_clips"], panels*a["synthetic_arms"]*24+360, "formal rating composition")
    for kind in ["pilot", "formal"]:
        hours = a[f"{kind}_rating_clips"]*a["raters"]*a["rating_seconds_assumed"]/3600
        close(e[f"{kind}_rating_person_hours_raw"], hours, f"{kind} rating hours")
    rows = e["scenarios"]
    require(len(rows) == len(a["generation_RTF_scenarios"]), "E3 scenario count mismatch")
    for row, rtf in zip(rows, a["generation_RTF_scenarios"]):
        generation = attempted*rtf
        total = (generation+qa+features+training)*m+a["setup_transfer_GPU_hours"]
        for key, expected in {"assumed_RTF":rtf, "generation_GPU_h":generation, "QA_GPU_h":qa,
                              "new_feature_GPU_h":features, "CNN_and_optional_FT_GPU_h":training,
                              "total_GPU_h":total, "GPU_USD":total*rate,
                              "continuous_single_GPU_days":total/24}.items():
            close(row[key], expected, f"E3 RTF={rtf} {key}")
    a, g = p["G"], b["G"]
    all_fits = math.prod(a[key] for key in ["strategies", "exposures", "blocks", "draws", "models"])
    close(g["all_fits"], all_fits, "G fits")
    close(g["CNN"], all_fits/2, "G CNN"); close(g["Ridge"], all_fits/2, "G Ridge")
    close(a["models"], 2, "G two models"); close(a["distinct_query_speakers"], a["blocks"]*12, "G distinct people")
    a, e4 = p["E4"], b["E4"]
    panels = a["train_language_conditions"]*a["speaker_folds"]*a["draws"]
    close(a["Ridge_seeds"], 1, "deterministic Ridge seed count")
    for key, expected in {"data_panels":panels, "Ridge_fits":panels*a["Ridge_seeds"],
                          "CNN_fits":panels*a["CNN_seeds"],
                          "total_fits":panels*(a["Ridge_seeds"]+a["CNN_seeds"])}.items():
        close(e4[key], expected, f"E4 {key}")
    require(e4["GPU_hours"] is None, "E4 has no priced GPU duration")
    auto = p["platform_comparison"]["autodl_5090_CNY_h"]
    close(b["AutoDL"]["E0_E1_E2_full_CNY"], [value*auto for value in b["E2"]["E0_E1_pilot_plus_full_GPU_hours"]], "AutoDL E2")
    require(len(b["AutoDL"]["E3_scenarios"]) == len(rows), "AutoDL scenario count mismatch")
    for actual, row in zip(b["AutoDL"]["E3_scenarios"], rows):
        close(actual["assumed_RTF"], row["assumed_RTF"], "AutoDL RTF")
        close(actual["GPU_CNY"], row["total_GPU_h"]*auto, "AutoDL E3 CNY")
    storage = p["storage"]
    monthly = storage["network_volume_GB"]*storage["USD_per_GB_month"]
    close(b["storage_USD"]["30_days"], monthly, "storage month")
    close(b["storage_USD"]["7_days"], monthly*7/storage["days_per_month_assumed"], "storage week")
    for row in b["precision_sensitivity"]:
        se = row["paired_SD_assumed_pp"]/math.sqrt(row["n_speakers"])
        close(row["approx_80pct_MDE_pp"], (NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.8))*se, "normal MDE")
        close(row["approx_CI_halfwidth_pp"], NormalDist().inv_cdf(.975)*se, "normal CI halfwidth")
    check_budget_markdown(root, b)
    return {"E2_core_fits": b["E2"]["core_fits"], "E3_core_fits": b["E3"]["core_fits"],
            "G_fits":g["all_fits"], "E4_fits":e4["total_fits"],
            "parameters_match_script_literal_without_execution":True}


def check_budget_markdown(root, b):
    p = b["parameters"]
    rows = table_rows((root / "budget_table.md").read_text(encoding="utf-8-sig"))
    for label, values in [("E2 核心540 fits", b["E2"]["core_GPU_hours"]),
                          ("E2 含180 FT", b["E2"]["with_FT_GPU_hours"])]:
        found = [row for row in rows if row[0] == label]
        require(len(found) == 1, f"missing/duplicate budget row: {label}")
        close(displayed_numbers(found[0][1]), [round(x, 2) for x in values], f"display {label}")
    for scenario in b["E3"]["scenarios"]:
        found = [row for row in rows if len(row) == 5 and row[0] == f"{scenario['assumed_RTF']:g}"]
        require(len(found) == 1, "missing/duplicate E3 budget display")
        expected = [scenario[key] for key in ["generation_GPU_h", "total_GPU_h", "GPU_USD", "continuous_single_GPU_days"]]
        close([float(x) for x in found[0][1:]], [round(x, 2) for x in expected], "E3 display")
    report = (root / "research_plan.md").read_text(encoding="utf-8-sig").replace("*", "").replace(",", "")
    match = re.search(r"(\d+)\s*Ridge\s*[＋+]\s*(\d+)\s*CNN\s*[＝=]\s*(\d+)\s*次拟合", report)
    require(match is not None, "missing explicit E4 fit breakdown in report")
    close([int(value) for value in match.groups()], [b["E4"][key] for key in ["Ridge_fits", "CNN_fits", "total_fits"]], "E4 report count")
    rating = re.search(r"24\s*人设计的纯标注约\s*(\d+)\s*[–-]\s*(\d+)\s*人小时.*?为\s*(\d+)\s*[–-]\s*(\d+)\s*人小时.*?40\s*人设计对应\s*(\d+)\s*[–-]\s*(\d+)\s*人小时", report, flags=re.S)
    require(rating is not None, "missing E4 human-hour ranges")
    raw = [24*2*40*4*3*sec/3600 for sec in (20, 30)]
    managed24 = [x*1.25 for x in raw]
    managed40 = [40*2*40*4*3*sec/3600*1.25 for sec in (20, 30)]
    close([int(x) for x in rating.groups()], [round(x) for x in raw+managed24+managed40], "E4 annotation hours")
    for phrase, key in [(r"纯听评共\s*(\d+)\s*人小时", "pilot_rating_person_hours_raw"),
                        (r"纯听评约\s*(\d+)\s*人小时", "formal_rating_person_hours_raw")]:
        match = re.search(phrase, report)
        require(match is not None, f"missing rating statement: {key}")
        close(int(match[1]), b["E3"][key], f"report {key}")
    comparison = (root / "cost_comparison_autodl.md").read_text(encoding="utf-8-sig")
    rows = table_rows(comparison)
    snapshot = read_json(root / "pricing_snapshot.json")
    prices = {item["name"].replace(" ", ""):item["price"] for page in snapshot["catalog"] for item in page["items"]}
    fx = p["platform_comparison"]["USD_to_CNY_budget_assumption"]
    require(fx > 0 and math.isfinite(fx), "FX assumption must be finite and positive")
    for gpu, auto in [("RTX4090",1.88), ("RTX5090",p["platform_comparison"]["autodl_5090_CNY_h"])]:
        found = [row for row in rows if row[0].startswith(gpu)]
        require(len(found) == 1, f"missing AutoDL rate row: {gpu}")
        expected = [auto, prices[gpu]["community"]*fx, prices[gpu]["secure"]*fx]
        close([float(row) for row in found[0][1:]], [round(x, 2) for x in expected], f"AutoDL FX {gpu}")
    found = [row for row in rows if row[0].startswith("E0/E1/E2")]
    require(len(found) == 1, "missing AutoDL full E2 comparison")
    hours = b["E2"]["E0_E1_pilot_plus_full_GPU_hours"]
    close(displayed_numbers(found[0][1]), [round(x*p["platform_comparison"]["autodl_5090_CNY_h"]) for x in hours], "AutoDL E2 display")
    close(displayed_numbers(found[0][2]), [round(x*p["hourly_rate_USD"]*fx) for x in hours], "Runpod FX E2 display")
    for row in rows:
        match = re.search(r"E3.*?RTF\s*=\s*([\d.]+)", row[0])
        if match:
            candidates = [s for s in b["E3"]["scenarios"] if s["assumed_RTF"] == float(match[1])]
            require(len(candidates) == 1, "unknown AutoDL displayed RTF")
            value = candidates[0]
            close(float(row[1]), round(value["total_GPU_h"]*p["platform_comparison"]["autodl_5090_CNY_h"]), "AutoDL E3 display")
            close(float(row[2]), round(value["GPU_USD"]*fx), "Runpod FX E3 display")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--preflight", action="store_true", help="skip missing/unsealed manifest; never reports sealed integrity")
    args = parser.parse_args(argv)
    root = args.packet_dir.resolve()
    results, errors = {}, []
    try:
        require(root.is_dir(), "packet directory missing")
        files = packet_files(root)
    except (OSError, ValueError) as exc:
        files = []; errors.append(f"inventory: {exc}")
    phases = [("Markdown_links", lambda: check_links(root, files, args.preflight)), ("budget_arithmetic", lambda: check_budget(root))]
    if not args.preflight:
        phases.insert(0, ("manifest_integrity", lambda: check_manifest(root, files)))
    for name, action in phases:
        try:
            results[name] = action()
        except (OSError, ValueError, KeyError, TypeError, IndexError, SyntaxError, ArithmeticError) as exc:
            errors.append(f"{name}: {exc}")
    status = "failed" if errors else ("preflight_pass" if args.preflight else "packet_integrity_and_arithmetic_pass")
    output = {"status":status, "manifest_checked":not args.preflight,
              "scientific_design_validated":False, "new_experiments_executed":False,
              "scope":"File inventory/SHA256, local link boundaries, and declared planning arithmetic only.",
              "checks":results, "errors":errors,
              "limits":["Remote links were not fetched; heading fragments not resolved.",
                        "Budget checks bind the current declared plan, not arbitrary future designs.",
                        "No authentication of historical raw receipts or statistical power certification."]}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
