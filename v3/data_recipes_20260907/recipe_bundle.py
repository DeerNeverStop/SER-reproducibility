"""Export the executed Study II manifests; stdlib only, no planning or fitting.

The pinned originals are the authority. Deduplicated lists are conveniences whose
ordered contents must reconstruct every original unit. This is a metadata-only
package, not a new preregistration, an audio distribution, or result recomputation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from collections import Counter, defaultdict


SCHEMA = "ser-study2-data-recipes-1"
PLAN_SHA256 = "f258cabe582d97b3a386a666b566730d00241a5ded0e14ad61d590cb3c861503"
PLAN_RAW_SHA256 = "27d42104e9df3b031c11293caec096f714eb4f13caedde8c9419fc517156ada9"
REP_RAW_SHA256 = "19f03f6cdd17b91e35b9385b56641908b21875b07219236f9e80f7a9ae7e463e"
EVIDENCE = "v3/data_design/evidence/"
RESULTS = "paper/study2/results/"
LOCK = EVIDENCE + "PLAN_LOCK.json"
PLAN = EVIDENCE + "core_plan.json.gz"
REP = EVIDENCE + "representative_manifest.csv.gz"
PAPER_MANIFEST = RESULTS + "BUNDLE_MANIFEST.json"
FIXED_SHA256 = {
    LOCK: "9861f122e12fabdea9d53a86f12461f9d8822990603baad1d906b207baa0fb2b",
    PLAN: "76a84b56189af5e2a9888f8690a912f1902f7db0eb7f51ede9f5bd4da179cf43",
    REP: "7d3f2eeed0e7993704e272a36006533365c79df61f2353b581622dfbaeea6740",
    "v2/manifests/cremad_manifest.csv": "e43224f032d474ac5e0d1b46d82cb7580c88bb08bac87e989fc767fd8eef552b",
    "v3/data_design/metadata/VideoDemographics.csv": "e33ddc8b60bedf98541735801127cbb67098b5957271adbdd5dea8a44294eec3",
    EVIDENCE + "unrepresented_cells.json": "5f8deefd017af811fd4d9ff81689e16be007a650197c82d37fa98939066ff643",
    EVIDENCE + "preflight_verification.json": "3292388b236aa3f4b264f07bcba8fb9b23ff95972c70b58f2ed438220370a1fd",
    "v3/data_design/PRIOR_EXPOSURE.md": "881d2e7df0bc731a2db50cf66b3411b02767f6e6f6bf31058845962e6e3f3603",
    PAPER_MANIFEST: "3b045f5d4eedb381b80ad9c78e801fe4c1a2d4f377ef8601dfb92bdedf3338f2",
}
EXECUTION_FILES = (
    "completion.json.gz", "analysis_lock.json.gz", "verification.json.gz",
    "merge_provenance.json.gz", "MERGE_DONE.gz", "ledger.jsonl.gz",
)
ROLES = ("fit", "val", "test")
REP_FIELDS = ("relative_path", "speaker", "sentence", "label_index", "intensity", "sha256", "selection_reason")
SCOPE = {
    "kind": "metadata_only_export_of_executed_Study_II",
    "new_preregistration": False,
    "new_plan_generated": False,
    "new_training_or_scientific_recomputation": False,
    "contains_audio_features_weights_or_predictions": False,
    "historical_receipts_only": True,
    "claim": "Exact recorded configurations; not an optimal recipe or proof of eliminating SD/SI gaps.",
    "history": "Original receipt absolute paths are preserved as historical evidence, not rebound to this bundle.",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def json_bytes(value):
    return canonical(value) + b"\n"


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key")
        obj[key] = value
    return obj


def parse_json(data):
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def safe_member(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name
            and all(ord(c) >= 32 for c in name), "unsafe relative member")
    p = PurePosixPath(name)
    require(not p.is_absolute() and all(x not in (".", "..", "") for x in name.split("/"))
            and p.as_posix() == name, "unsafe relative member")
    return name


def _no_links(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if os.path.lexists(part):
            s = part.lstat()
            require(not stat.S_ISLNK(s.st_mode) and not
                    (getattr(s, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)),
                    "symlink/reparse path rejected")
    return path


def member_path(root, name):
    safe_member(name)
    root = _no_links(root)
    target = _no_links(root.joinpath(*name.split("/")))
    require(target.is_relative_to(root), "member escapes root")
    return target


def _read(root, name):
    path = member_path(root, name)
    require(path.is_file(), "missing file: " + name)
    return path.read_bytes()


def _files(root):
    root = _no_links(root)
    require(root.is_dir(), "bundle is not a directory")
    result = set()
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = _no_links(Path(current) / name)
            rel = safe_member(p.relative_to(root).as_posix())
            if name in files:
                require(p.is_file(), "nonregular bundle member")
                result.add(rel)
    return result


def _new_output(out, protected):
    out = _no_links(out)
    for source in protected:
        source = _no_links(source)
        require(not out.is_relative_to(source) and not source.is_relative_to(out),
                "output overlaps input tree")
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),
            "output must be absent or an empty directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(root, name, data):
    path = member_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def _originals(root, prefix=""):
    """Read the exact fixed original files; no code from them is executed."""
    blobs = {}
    for name, expected in FIXED_SHA256.items():
        blobs[name] = _read(root, prefix + name)
        require(sha256(blobs[name]) == expected, "pinned original changed: " + name)
    lock = parse_json(blobs[LOCK])
    require(len(lock["source_sha256"]) == 16, "source closure size")
    for name, expected in lock["source_sha256"].items():
        blobs[name] = _read(root, prefix + safe_member(name))
        require(sha256(blobs[name]) == expected, "frozen source changed: " + name)
    paper = parse_json(blobs[PAPER_MANIFEST])
    for name in EXECUTION_FILES:
        key = RESULTS + name
        blobs[key] = _read(root, prefix + key)
        require(sha256(blobs[key]) == paper["files_sha256"][name], "execution receipt changed")
        raw = gzip.decompress(blobs[key])
        pin = paper["compressed_sources"][name]
        require(len(raw) == pin["uncompressed_bytes"] and sha256(raw) == pin["source_sha256"],
                "decompressed receipt changed")
    plan_raw = gzip.decompress(blobs[PLAN])
    require(sha256(plan_raw) == PLAN_RAW_SHA256, "original plan bytes changed")
    plan = parse_json(plan_raw)
    require(plan["plan_sha256"] == PLAN_SHA256 == lock["plan_sha256"]
            and sha256(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})) == PLAN_SHA256,
            "original plan identity mismatch")
    require(plan["input"] == lock["input"] and plan["source_sha256"] == lock["source_sha256"],
            "plan lock mismatch")
    rep_raw = gzip.decompress(blobs[REP])
    require(sha256(rep_raw) == REP_RAW_SHA256, "representative CSV bytes changed")
    reps = _csv(rep_raw)
    raw = _csv(blobs[plan["input"]["manifest_path"]])
    demo = _csv(blobs[plan["input"]["demographics_path"]])
    sex = {row["ActorID"]: row["Sex"] for row in demo}
    raw_by_path = {r["relative_path"]: r for r in raw}
    require(len(raw) == len(raw_by_path) == 7442 and len(reps) == 6524, "metadata size")
    rep = {r["relative_path"]: r for r in reps}
    require(len(rep) == 6524 and list(reps[0]) == list(REP_FIELDS), "representative schema")
    groups = defaultdict(list)
    for row in raw:
        groups[row["sha256"]].append(row)
    removed = {"1076_MTI_SAD_XX.wav"}
    for group in groups.values():
        drop = group if len({r["label"] for r in group}) > 1 else sorted(group, key=lambda r: r["relative_path"])[1:]
        removed.update(r["relative_path"] for r in drop)
    clean = [r for r in raw if r["relative_path"] not in removed]
    require(len(clean) == 7435, "clean metadata size")
    cells = defaultdict(list)
    for row in clean:
        if row["intensity"] in ("MD", "XX"):
            cells[(row["speaker"], row["sentence"], row["label_index"])].append(row)
    selected = {min(v, key=lambda r: (r["intensity"] != "MD", r["relative_path"]))["relative_path"]
                for v in cells.values()}
    require(selected == set(rep), "saved representative selection differs from pinned metadata rule")
    for path, row in rep.items():
        safe_member(path)
        require(all(row[k] == raw_by_path[path][k] for k in REP_FIELDS if k != "selection_reason"),
                "representative join differs")
        require(sex.get(row["speaker"]) in ("Female", "Male"), "missing sex stratum")
        require(row["selection_reason"] == ("MD_preferred" if row["intensity"] == "MD" else "XX_no_MD"),
                "representative reason differs")
    units = plan["units"]
    require(len(units) == len({u["unit_id"] for u in units}) == 1440, "unit count")
    for unit in units:
        require(sha256(canonical({k: v for k, v in unit.items() if k != "unit_id"})) == unit["unit_id"],
                "original unit identity differs")
        for role in ROLES:
            require(len(unit[role]) == len(set(unit[role])) and set(unit[role]) <= set(rep),
                    "unit paths invalid")
    comp = parse_json(gzip.decompress(blobs[RESULTS + "completion.json.gz"]))
    alock = parse_json(gzip.decompress(blobs[RESULTS + "analysis_lock.json.gz"]))
    require(comp["plan_sha256"] == alock["plan_sha256"] == PLAN_SHA256
            and comp["n_units"] == comp["units_done"] == alock["n_units"] == 1440
            and set(comp["done_sha256"]) == {u["unit_id"] for u in units}
            and comp["done_sha256"] == alock["done_sha256"]
            and alock["completion_sha256"] == sha256(gzip.decompress(blobs[RESULTS + "completion.json.gz"])),
            "historical execution identity does not close")
    return blobs, plan, rep, sex


def _derive(plan):
    components, index = {}, []
    for unit in plan["units"]:
        item = {k: v for k, v in unit.items() if k not in ROLES}
        item["components"] = {}
        for role in ROLES:
            paths = unit[role]
            name = f"components/{role}/{sha256(canonical(paths))}.json"
            data = json_bytes(paths)
            require(name not in components or components[name] == data, "component hash collision")
            components[name] = data
            item["components"][role] = name
        index.append(item)
    counts = {"units": len(index), "representative_rows": 6524,
              **{role + "_components": sum(x.startswith(f"components/{role}/") for x in components) for role in ROLES}}
    require(counts == {"units": 1440, "representative_rows": 6524, "fit_components": 720,
                       "val_components": 90, "test_components": 90}, "unexpected component counts")
    return components, {"schema": SCHEMA, "units": index}, counts


def _load_bundle(bundle):
    bundle = _no_links(bundle)
    manifest_raw = _read(bundle, "manifest.json")
    manifest = parse_json(manifest_raw)
    require(manifest["schema"] == SCHEMA and manifest["scope"] == SCOPE
            and manifest["original_plan_sha256"] == PLAN_SHA256, "bundle identity/scope mismatch")
    files = manifest["files"]
    require(isinstance(files, dict) and len(files) < 2000, "invalid inventory")
    for name in files:
        safe_member(name)
    require(_files(bundle) == set(files) | {"manifest.json"}, "unlisted/missing bundle payload")
    for name, receipt in files.items():
        data = _read(bundle, name)
        require(receipt == {"bytes": len(data), "sha256": sha256(data)}, "payload changed: " + name)
    originals, plan, rep, sex = _originals(bundle, "originals/")
    components, expected_index, counts = _derive(plan)
    expected_names = {"originals/" + k for k in originals} | set(components) | {"units.json"}
    require(set(files) == expected_names and manifest["original_files"] == sorted(originals)
            and manifest["counts"] == counts, "bundle contents/counts differ from original plan")
    index = parse_json(_read(bundle, "units.json"))
    require(index == expected_index, "unit index differs from original plan")
    for name, data in components.items():
        require(_read(bundle, name) == data, "ordered component differs from original")
    for original, item in zip(plan["units"], index["units"]):
        rebuilt = {k: v for k, v in item.items() if k != "components"}
        for role in ROLES:
            rebuilt[role] = parse_json(_read(bundle, item["components"][role]))
        require(rebuilt == original and sha256(canonical({k: v for k, v in rebuilt.items() if k != "unit_id"}))
                == original["unit_id"], "lossless unit reconstruction failed")
    receipt = {"pass": True, "schema": SCHEMA, "plan_sha256": PLAN_SHA256,
               "manifest_sha256": sha256(manifest_raw), "counts": counts,
               "payload_files": len(files), "payload_bytes": sum(x["bytes"] for x in files.values()),
               "scope": SCOPE}
    return receipt, plan, rep, sex


def build(repo, out):
    """Copy pinned originals and losslessly index them; return verification receipt."""
    repo = _no_links(repo)
    originals, plan, _, _ = _originals(repo)
    components, index, counts = _derive(plan)
    payload = {"originals/" + name: blob for name, blob in originals.items()}
    payload.update(components)
    payload["units.json"] = json_bytes(index)
    out = _new_output(out, (repo,))
    for name, data in payload.items():
        _write(out, name, data)
    manifest = {"schema": SCHEMA, "scope": SCOPE, "original_plan_sha256": PLAN_SHA256,
                "original_files": sorted(originals), "counts": counts,
                "files": {name: {"bytes": len(data), "sha256": sha256(data)}
                          for name, data in sorted(payload.items())}}
    _write(out, "manifest.json", json_bytes(manifest))
    return verify(out)


def verify(bundle):
    """Verify bytes, original pins, metadata joins and all 1,440 reconstructed units."""
    return _load_bundle(bundle)[0]


def export_unit(bundle, unit_id, out):
    """Export one original unit in original row order, with no audio conversion."""
    require(isinstance(unit_id, str) and re.fullmatch(r"[0-9a-f]{64}", unit_id), "invalid unit_id")
    receipt, plan, rep, sex = _load_bundle(bundle)
    matched = [u for u in plan["units"] if u["unit_id"] == unit_id]
    require(len(matched) == 1, "unknown original unit_id")
    unit = matched[0]
    payload = {"unit.json": json_bytes(unit)}
    for role in ROLES:
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=("row_index",) + REP_FIELDS + ("sex",), lineterminator="\n")
        writer.writeheader()
        for index, path in enumerate(unit[role]):
            row = rep[path]
            writer.writerow({"row_index": index, **row, "sex": sex[row["speaker"]]})
        payload[role + ".csv"] = text.getvalue().encode("utf-8")
    out = _new_output(out, (bundle,))
    for name, data in payload.items():
        _write(out, name, data)
    result = {"pass": True, "schema": SCHEMA + "-unit-export", "scope": SCOPE,
              "unit_id": unit_id, "plan_sha256": PLAN_SHA256,
              "bundle_manifest_sha256": receipt["manifest_sha256"],
              "row_index": "zero-based; original ordered fit/val/test paths",
              "val_role": "early-stop validation; not final test",
              "sex_field": "joined from pinned official demographics; original representative CSV unchanged",
              "rows": {role: len(unit[role]) for role in ROLES},
              "files": {name: {"bytes": len(data), "sha256": sha256(data)} for name, data in payload.items()}}
    _write(out, "export_receipt.json", json_bytes(result))
    for name, data in payload.items():
        require(_read(out, name) == data, "export readback mismatch")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("verify")
    p.add_argument("--bundle", type=Path, required=True)
    p = commands.add_parser("export")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--unit-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.repo, args.out)
        elif args.command == "verify":
            result = verify(args.bundle)
        else:
            result = export_unit(args.bundle, args.unit_id, args.out)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, "recipe bundle rejected: " + str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
