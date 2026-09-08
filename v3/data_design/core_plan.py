"""Outcome-blind, exact-budget CREMA-D core and pilot plans (standard library)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ranked(items, salt):
    return sorted(items, key=lambda x: (digest([salt, x]), x))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def read_metadata(repo):
    manifest = repo / "v2/manifests/cremad_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    require(len(raw) == 7442, "unexpected pinned corpus size")
    require(len({r['relative_path'] for r in raw}) == len(raw), "duplicate path")
    by_hash = defaultdict(list)
    for row in raw:
        by_hash[row["sha256"]].append(row)
    removed = set()
    for group in by_hash.values():
        if len({r["label"] for r in group}) > 1:
            removed.update(r["relative_path"] for r in group)
        else:
            removed.update(r["relative_path"] for r in sorted(group, key=lambda r: r["relative_path"])[1:])
    removed.add("1076_MTI_SAD_XX.wav")
    clean = sorted((r for r in raw if r["relative_path"] not in removed), key=lambda r: r["relative_path"])
    require(len(clean) == 7435, "hygienic population mismatch")
    groups = defaultdict(list)
    for row in clean:
        groups[(row["speaker"], row["sentence"], int(row["label_index"]))].append(row)
    cells = {}
    for key, group in groups.items():
        options = [r for r in group if r["intensity"] in ("MD", "XX")]
        if options:
            cells[key] = min(options, key=lambda r: (0 if r["intensity"] == "MD" else 1, r["relative_path"]))
    demo_path = repo / "v3/data_design/metadata/VideoDemographics.csv"
    with demo_path.open(encoding="utf-8-sig", newline="") as handle:
        demo = {r["ActorID"]: r["Sex"] for r in csv.DictReader(handle)}
    speakers = sorted({r["speaker"] for r in clean})
    require(all(demo.get(s) in ("Male", "Female") for s in speakers), "missing source sex stratum")
    return {"raw": raw, "clean": clean, "cells": cells, "speakers": speakers,
            "prompts": sorted({r["sentence"] for r in clean}), "sex": demo,
            "input": {"manifest_path": manifest.relative_to(repo).as_posix(),
                      "manifest_sha256": file_sha(manifest),
                      "demographics_path": demo_path.relative_to(repo).as_posix(),
                      "demographics_sha256": file_sha(demo_path)}}


def roles(meta, draw, fold):
    test, stop = set(), set()
    for sex in ("Female", "Male"):
        group = [s for s in meta["speakers"] if meta["sex"][s] == sex]
        test.update(ranked(group, f"core:{draw}:{sex}:outer")[fold::5])
    for sex in ("Female", "Male"):
        available = [s for s in meta["speakers"] if s not in test and meta["sex"][s] == sex]
        stop.update(ranked(available, f"core:{draw}:{fold}:{sex}:stop")[:4])
    return test, stop, set(meta["speakers"]) - test - stop


def prompt_slots(meta, rotation):
    order = ranked(meta["prompts"], "core:prompt-pairs:20260905")
    test = order[2 * rotation:2 * rotation + 2]
    remaining = ranked(set(order) - set(test), f"core:prompt-common:{rotation}")
    common, replacement = remaining[:6], remaining[6:8]
    return test, common, {"prompt_seen": common + test, "prompt_new": common + replacement}


def rows_for(meta, speakers, prompts):
    rows = []
    for speaker in sorted(speakers):
        for label in range(6):
            found = [meta["cells"][(speaker, prompt, label)] for prompt in prompts
                     if (speaker, prompt, label) in meta["cells"]]
            require(found, f"evaluation lacks class {speaker}:{label}")
            rows.extend(found)
    return sorted(rows, key=lambda r: r["relative_path"])


def select_speakers(meta, pool, prompts, salt, count=48):
    require(isinstance(count, int) and count > 0 and count % 2 == 0,
            "speaker count must be a positive even integer")
    # Complete cells on the union support exactly the same speaker/slot graph
    # under both sentence identities; exclusions are recorded in the inventory.
    eligible = [s for s in pool if all((s, p, c) in meta["cells"] for p in prompts for c in range(6))]
    females = ranked([s for s in eligible if meta["sex"][s] == "Female"], salt + ":F")
    males = ranked([s for s in eligible if meta["sex"][s] == "Male"], salt + ":M")
    # Fixed equal metadata-sex proportions across S=12/48.
    nf = count // 2
    require(len(females) >= nf and len(males) >= count - nf, f"insufficient complete eligible people: {salt}")
    selected = []
    for i in range(nf):
        selected.extend([females[i], males[i]])
    return selected, eligible


def panel(meta, speaker_order, slots, budget, salt, stratified=False):
    count = len(speaker_order)
    require(count > 0 and count == len(set(speaker_order)), "empty or duplicate speakers")
    require(len(slots) == len(set(slots)) == 8, "exactly eight unique prompt slots required")
    require(budget % (6 * count) == 0, "fractional per-person prompt quota")
    k = budget // (6 * count)
    require(1 <= k <= 8, "prompt quota exceeds available slots")
    # Randomize identities but use the same graph in seen/new. All columns
    # have exactly B/48 speakers, avoiding a one-person definition of seen.
    slot_order = ranked(list(range(8)), salt + ":slots")
    rows = []
    within_sex = Counter()
    for i, speaker in enumerate(speaker_order):
        if stratified:
            sex = meta['sex'][speaker]
            i = within_sex[sex]
            within_sex[sex] += 1
        indices = [slot_order[(i*k + j) % 8] for j in range(k)]
        for index in indices:
            rows.extend(meta["cells"][(speaker, slots[index], c)] for c in range(6))
    require(len(rows) == budget == len({r["sha256"] for r in rows}), "budget/byte identity failed")
    require(set(Counter(r["sentence"] for r in rows).values()) == {budget // 8}, "unequal prompt exposure")
    return sorted(r["relative_path"] for r in rows)


def feature_identity(features):
    identities, names = {}, {}
    for kind in ("wavlm_base_plus", "logmel"):
        paths = list(features.glob(f"cremad__{kind}__*.npz"))
        require(len(paths) == 1, f"cache missing or ambiguous: {kind}")
        path = paths[0]
        require("SYNTHETIC" not in path.name.upper(), "synthetic cache")
        side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        sha = file_sha(path)
        require(side["sha256"] == sha and side["corpus"] == "cremad", "cache sidecar mismatch")
        if kind != "logmel":
            require(side["encoder"] == kind, "cache encoder mismatch")
        identities[kind], names[kind] = sha, path.name
    return {"feature_sha256": identities, "feature_files": names}


def build(repo, features, require_runtime=True):
    meta = read_metadata(repo)
    inputs = {**meta["input"], **feature_identity(features)}
    units, feasibility = [], []
    for draw in range(3):
        for fold in range(5):
            test, stop, pool = roles(meta, draw, fold)
            for rotation in range(6):
                test_prompts, common, slots = prompt_slots(meta, rotation)
                salt = f"core:{draw}:{fold}:{rotation}"
                ordered, eligible = select_speakers(meta, pool, set().union(*map(set, slots.values())), salt)
                test_paths = [r["relative_path"] for r in rows_for(meta, test, test_prompts)]
                val_paths = [r["relative_path"] for r in rows_for(meta, stop, common)]
                feasibility.append({"draw": draw, "fold": fold, "rotation": rotation,
                                    "eligible": len(eligible), "train_pool": len(pool),
                                    "eligible_female": sum(meta["sex"][s] == "Female" for s in eligible),
                                    "test_speakers": len(test), "test_rows": len(test_paths),
                                    "val_rows": len(val_paths)})
                for budget in (288, 576):
                    for count in (12, 48):
                        for scenario, prompts in slots.items():
                            fit = panel(meta, ordered[:count], prompts, budget, salt, stratified=True)
                            seed = int(digest(["train", draw, fold, rotation])[:8], 16) % (2**31 - 1)
                            for model in ("ridge_wavlm", "cnn"):
                                config = ({"model": "ridge_a1_wavlm_base_plus", "feature_state": 12,
                                           "alpha": 1.0, "class_weight": "balanced", "solver": "lsqr", "tol": 0.0001}
                                          if model == "ridge_wavlm" else
                                          {"model": "cnn", "batch_size": 32, "epochs": 100, "patience": 15,
                                           "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1})
                                unit = {"model": model, "fold": fold, "rotation": rotation, "draw": draw,
                                        "scenario": scenario, "B": budget, "S": count, "P_global": 8,
                                        "P_per_speaker": budget // (6 * count), "R": 1,
                                        "train_seed": seed, "fit": fit, "val": val_paths,
                                        "test": test_paths, "config": config}
                                unit["unit_id"] = digest(unit)
                                units.append(unit)
    require(len(units) == len({u["unit_id"] for u in units}) == 1440, "unexpected plan size")
    source_paths = ["v2/ser_v2/train.py", "v2/ser_v2/corpora.py", "v2/ser_v2/features.py",
                    "v2/ser_v2/common.py", "v2/ser_v2/stats.py", "advanced_models.py",
                    "tuned_standard_experiment.py", "advanced_experiment_utils.py",
                    "experiment_utils.py", "features.py", "dataset.py", "fno_model.py",
                    "v3/data_design/core_plan.py", "v3/data_design/core_run.py",
                    "v3/data_design/core_verify.py", "v3/data_design/SPEC_CORE_ZH.md"]
    if require_runtime:
        require(all((repo / p).is_file() for p in source_paths), "runtime/verification/spec not ready for lock")
    sources = {p: file_sha(repo / p) for p in source_paths if (repo / p).is_file()}
    design = {"status": "prospectively_specified_estimation_not_new_confirmatory_evidence",
              "prior_exposure": "v2 results and Claude review informed design; see PRIOR_EXPOSURE.md",
              "core_question": "fixed-budget speaker-count versus per-person prompt-coverage tradeoff",
              "representative": "MD then XX; exclude cell without either; no HI or LO substitution",
              "sex_source": "official CREMA-D VideoDemographics.csv; Female/Male metadata strata",
              "training_sex_ratio": "equal Female/Male for both S, also balanced within every prompt",
              "speaker_folds": 5, "prompt_rotations": 6, "draws": 3,
              "budgets": [288, 576], "speakers": [12, 48], "global_prompts": 8,
              "prompt_coverage_speakers": {"288": 6, "576": 12},
              "stop_speakers": 8, "stop_prompts": "common six in both scenarios",
              "primary_contrasts": "S48-S12 under prompt_new separately at each B, per model",
              "sesoi_pp": 3.0, "sesoi_status": "planning threshold, not derived from MDE",
              "no_selection": True, "scientific_scores_hidden_until_complete": True,
              "runtime_complete": require_runtime}
    plan = {"schema": "ser-study2-core-1", "program": "SER26-STUDY2-CORE-1", "input": inputs,
            "source_sha256": sources, "design": design, "units": units}
    plan["plan_sha256"] = digest(plan)
    inventory = {"schema": "ser-study2-capacity-1", "plan_sha256": plan["plan_sha256"],
                 "raw_rows": len(meta["raw"]), "clean_rows": len(meta["clean"]),
                 "representative_cells": len(meta["cells"]),
                 "representative_intensity": dict(Counter(r["intensity"] for r in meta["cells"].values())),
                 "core_units": len(units), "by_model": dict(Counter(u["model"] for u in units)),
                 "conditions": feasibility}
    return plan, inventory, meta


def pilot_from(meta, inputs):
    # Engineering-only subset with a separate held-out selection population;
    # final core scientific scores remain unread. No test paths in this object.
    units = []
    test, stop, pool = roles(meta, 0, 0)
    for sex in ("Female", "Male"):
        group = [s for s in pool if meta["sex"][s] == sex]
        # Fixed four per sex, kept out of pilot training.
        chosen = set(ranked(group, f"pilot:select:{sex}")[:4])
        if sex == "Female":
            select = chosen
        else:
            select |= chosen
    pool -= select
    order = ranked(meta["prompts"], "pilot:prompts")
    p_test, p_select, p_train = order[:2], order[2:4], order[4:]
    selection = [r["relative_path"] for r in rows_for(meta, select, p_select)]
    for draw in range(3):
        salt = f"pilot:{draw}"
        # Pilot S48 may have fewer women than the core after the extra holdout;
        # use a separate deterministic ordering, without claiming stratified effect.
        eligible = [s for s in pool if all((s,p,c) in meta["cells"] for p in p_train for c in range(6))]
        speaker_order = ranked(eligible, salt + ":speakers")
        require(len(speaker_order) >= 48, "pilot S48 is not feasible")
        for count in (12, 24, 48):
            fit = panel(meta, speaker_order[:count], p_train, 288, salt)
            unit = {"fit": fit, "select": selection, "train_seed": 20260905 + draw,
                    "config": {"B": 288, "S": count, "P_global": 8, "R": 1,
                               "P_per_speaker": 288 // (6*count)}}
            unit["unit_id"] = digest(unit)
            units.append(unit)
    inp = {k: inputs[k] for k in ("manifest_path", "manifest_sha256")}
    inp.update(feature_kind="wavlm_base_plus", feature_state=12,
               feature_sha256=inputs["feature_sha256"]["wavlm_base_plus"])
    plan = {"schema": "ser-study2-blind-pilot-1", "input": inp, "units": units}
    plan["plan_sha256"] = digest(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draft", action="store_true", help="metadata only; cannot authorize core execution")
    args = parser.parse_args()
    plan, inventory, meta = build(args.repo.resolve(), args.features, not args.draft)
    write_json(args.out / "core_plan.json", plan)
    write_json(args.out / "capacity.json", inventory)
    write_json(args.out / "pilot_plan.json", pilot_from(meta, plan["input"]))
    columns = ["relative_path", "speaker", "sentence", "label_index", "intensity", "sha256", "selection_reason"]
    with (args.out / "representative_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in sorted(meta["cells"].values(), key=lambda r: r["relative_path"]):
            writer.writerow({**{k: row[k] for k in columns[:-1]},
                             "selection_reason": "MD_preferred" if row["intensity"] == "MD" else "XX_no_MD"})
    unrepresented = [r for r in meta["clean"]
                     if (r["speaker"], r["sentence"], int(r["label_index"])) not in meta["cells"]]
    write_json(args.out / "unrepresented_cells.json", {"reason": "no_MD_or_XX_after_hygiene",
                                                        "records": unrepresented})
    print(canonical({"pass": True, "units": len(plan["units"]), "plan_sha256": plan["plan_sha256"],
                     "minimum_eligible": min(r["eligible"] for r in inventory["conditions"]),
                     "runtime_complete": not args.draft}))


if __name__ == "__main__":
    main()
