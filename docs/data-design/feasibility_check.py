"""Metadata-only constructive audit for the proposal; never trains or scores models.

Run from any checkout with the committed v2 manifests:
  python docs/data-design/feasibility_check.py --out snapshot.json
  python docs/data-design/feasibility_check.py --check docs/data-design/feasibility_snapshot.json

This deterministic demonstration is NOT the future experiment planner. It reads
public CSV metadata, independently applies the documented hygiene rules, and
constructs example panels. It does not read audio, caches, predictions or ledgers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ranked(items, salt):
    return sorted(items, key=lambda item: (digest([salt, item]), item))


def hygiene(corpus, rows):
    groups = defaultdict(list)
    for row in rows:
        require(row["sha256"] and len(row["sha256"]) == 64, "missing byte identity")
        groups[row["sha256"]].append(row)
    excluded = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        if len({row["label"] for row in group}) > 1:
            for row in group:
                excluded[row["relative_path"]] = "H1"
        else:
            for row in sorted(group, key=lambda row: row["relative_path"])[1:]:
                excluded[row["relative_path"]] = "H2"
    if corpus == "cremad":
        for row in rows:
            if row["relative_path"] == "1076_MTI_SAD_XX.wav":
                excluded[row["relative_path"]] = "H3"
    clean = sorted((dict(row) for row in rows if row["relative_path"] not in excluded),
                   key=lambda row: row["relative_path"])
    return clean, dict(sorted(Counter(excluded.values()).items()))


def load_corpus(repo, name):
    path = repo / "v2" / "manifests" / f"{name}_manifest.csv"
    require(path.is_file(), f"real manifest missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    require(raw and len({row["relative_path"] for row in raw}) == len(raw),
            "empty manifest or duplicate paths")
    rows, excluded = hygiene(name, raw)
    cells = defaultdict(list)
    for row in rows:
        cells[(row["speaker"], row["sentence"], row["label"])].append(row)
    # One fixed representative for CREMA-D; intensity is never counted as a take.
    for key in cells:
        cells[key] = sorted(cells[key], key=lambda row: row["relative_path"])
        if name == "cremad":
            cells[key] = cells[key][:1]
        else:
            require(len({row["take"] for row in cells[key]}) == len(cells[key]),
                    "SUBESCO take IDs are not unique within cell")
    return {
        "name": name, "cells": cells, "rows": rows,
        "speakers": sorted({row["speaker"] for row in rows}),
        "prompts": sorted({row["sentence"] for row in rows}),
        "labels": sorted({row["label"] for row in rows}),
        "identity": {
            "path": path.relative_to(repo).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_rows": len(raw), "clean_rows": len(rows),
            "excluded_by_rule": excluded, "unique_cells": len(cells),
            "speakers": len({row["speaker"] for row in rows}),
            "prompts": len({row["sentence"] for row in rows}),
            "classes": len({row["label"] for row in rows}),
        },
    }


def split_roles(corpus, fold):
    name = corpus["name"]
    order = ranked(corpus["speakers"], f"{name}:outer")
    test = set(order[fold::5])
    rest = ranked(set(order) - test, f"{name}:{fold}:inner")
    n_dev = 8 if name == "cremad" else 2
    stop, select = set(rest[:n_dev]), set(rest[n_dev:2 * n_dev])
    fitpool = set(rest[2 * n_dev:])
    prompt_order = ranked(corpus["prompts"], f"{name}:{fold}:prompts")
    p_test, p_select = set(prompt_order[:2]), set(prompt_order[2:4])
    require(not (test & stop or test & select or test & fitpool or stop & select or
                 stop & fitpool or select & fitpool), "speaker roles overlap")
    return test, stop, select, fitpool, p_test, p_select, prompt_order


def evaluation_rows(corpus, speakers, prompts):
    output = []
    for speaker in sorted(speakers):
        for label in corpus["labels"]:
            group = [corpus["cells"][(speaker, prompt, label)][0]
                     for prompt in sorted(prompts)
                     if (speaker, prompt, label) in corpus["cells"]]
            require(group, "evaluation speaker lacks a required class")
            output.extend(group)
    return output


def available_prompts(corpus, speaker, prompts, repeats):
    return {prompt for prompt in prompts
            if all(len(corpus["cells"].get((speaker, prompt, label), [])) >= repeats
                   for label in corpus["labels"])}


def construct(corpus, chosen_speakers, prompts, prompts_per_speaker, repeats, salt):
    available = {s: available_prompts(corpus, s, prompts, repeats)
                 for s in chosen_speakers}
    assignments = {s: set() for s in chosen_speakers}
    # Cover every global prompt without changing each speaker's exact quota.
    for prompt in ranked(prompts, salt + ":coverage"):
        possible = [s for s in chosen_speakers if prompt in available[s]
                    and len(assignments[s]) < prompts_per_speaker]
        require(possible, "global prompt coverage is infeasible")
        speaker = ranked(possible, salt + ":anchor:" + prompt)[0]
        assignments[speaker].add(prompt)
    for speaker in chosen_speakers:
        needed = prompts_per_speaker - len(assignments[speaker])
        assignments[speaker].update(ranked(available[speaker] - assignments[speaker],
                                          salt + ":fill:" + speaker)[:needed])
        require(len(assignments[speaker]) == prompts_per_speaker, "prompt quota failed")
    panel = []
    for speaker in chosen_speakers:
        for prompt in sorted(assignments[speaker]):
            for label in corpus["labels"]:
                group = corpus["cells"][(speaker, prompt, label)]
                chosen_paths = ranked([row["relative_path"] for row in group], salt + ":take")[:repeats]
                panel.extend(row for row in group if row["relative_path"] in chosen_paths)
    return panel


def audit(repo):
    inputs, records = {}, []
    for name in ("cremad", "subesco"):
        corpus = load_corpus(repo, name)
        inputs[name] = corpus["identity"]
        n_labels = len(corpus["labels"])
        for fold in range(5):
            test, stop, select, fitpool, p_test, p_select, p_order = split_roles(corpus, fold)
            test_rows = evaluation_rows(corpus, test, p_test)
            select_rows = evaluation_rows(corpus, select, p_select)
            for stage in ("candidate", "final"):
                pool = fitpool if stage == "candidate" else fitpool | select
                for draw in range(3):
                    salt = f"{name}:{fold}:{stage}:{draw}"
                    if name == "cremad":
                        new_allowed = set(p_order) - p_test - (p_select if stage == "candidate" else set())
                        new_prompts = set(ranked(new_allowed, salt + ":new")[:8])
                        seen_prompts = p_test | p_select | set(ranked(set(p_order) - p_test - p_select,
                                                                   salt + ":seen")[:4])
                        scenarios = {"prompt_seen": seen_prompts, "prompt_new": new_prompts}
                        configurations = [(b, s, 8, 1, b // (n_labels * s))
                                          for b in (288, 576) for s in (12, 24, 48)]
                    else:
                        allowed = set(p_order) - p_test - (p_select if stage == "candidate" else set())
                        scenarios = {"prompt_new": set(ranked(allowed, salt + ":new")[:4])}
                        configurations = [(224 * m, s, p, r * m, p)
                                          for m in (1, 2) for s, p, r in ((8, 4, 1), (4, 4, 2), (8, 2, 2))]
                    for budget, n_speakers, n_prompts, repeats, per_speaker in configurations:
                        scenario_prompts = {key: set(ranked(value, salt + ":subset")[:n_prompts])
                                            for key, value in scenarios.items()}
                        # Common eligible speakers make the CREMA condition pairing stronger.
                        eligible = [s for s in pool if all(
                            len(available_prompts(corpus, s, prompts, repeats)) >= per_speaker
                            for prompts in scenario_prompts.values())]
                        chosen = ranked(eligible, salt + ":speakers")[:n_speakers]
                        require(len(chosen) == n_speakers, f"insufficient eligible speakers: {salt}")
                        for scenario, prompts in scenario_prompts.items():
                            require(len(prompts) == n_prompts, "global prompt quota failed")
                            panel = construct(corpus, chosen, prompts, per_speaker, repeats, salt)
                            paths = [row["relative_path"] for row in panel]
                            hashes = [row["sha256"] for row in panel]
                            require(len(panel) == budget == len(set(paths)) == len(set(hashes)),
                                    "budget or duplicate constraint failed")
                            require(len({row["speaker"] for row in panel}) == n_speakers, "speaker quota failed")
                            require({row["sentence"] for row in panel} == prompts, "prompt coverage failed")
                            require(set(chosen).isdisjoint(test | stop), "final speaker leakage")
                            if stage == "candidate":
                                require(set(chosen).isdisjoint(select), "selection speaker leakage")
                            if scenario == "prompt_new":
                                require(prompts.isdisjoint(p_test), "test prompt leakage")
                                if stage == "candidate":
                                    require(prompts.isdisjoint(p_select), "selection prompt leakage")
                            else:
                                require(p_test <= prompts and p_select <= prompts, "seen condition not realized")
                            counts = Counter(row["label"] for row in panel)
                            require(set(counts) == set(corpus["labels"])
                                    and set(counts.values()) == {budget // n_labels}, "class quota failed")
                            # Early stopping must not change with SUBESCO's P/R policy.
                            stop_prompts = prompts if name == "cremad" else allowed
                            stop_rows = evaluation_rows(corpus, stop, stop_prompts)
                            require(set(paths).isdisjoint(row["relative_path"] for row in test_rows + stop_rows),
                                    "evaluation overlap")
                            if stage == "candidate":
                                require(set(paths).isdisjoint(row["relative_path"] for row in select_rows),
                                        "selection overlap")
                            records.append({"corpus": name, "fold": fold, "stage": stage,
                                            "draw": draw, "scenario": scenario, "B": budget,
                                            "S": n_speakers, "P_global": n_prompts,
                                            "P_per_speaker": per_speaker, "R": repeats,
                                            "panel_sha256": digest(sorted(paths)),
                                            "test_sha256": digest(sorted(row["relative_path"] for row in test_rows)),
                                            "stop_sha256": digest(sorted(row["relative_path"] for row in stop_rows)),
                                            "select_sha256": digest(sorted(row["relative_path"] for row in select_rows))})
    summary = Counter(row["corpus"] for row in records)
    require(summary == {"cremad": 360, "subesco": 180}, "unexpected audit scope")
    return {"schema": "ser-data-design-metadata-feasibility-1", "scope": "metadata-only; no training or scoring",
            "inputs": inputs, "checks": {"panel_constructions": len(records), "by_corpus": dict(summary),
            "outer_folds": 5, "subset_draws": 3, "stages": ["candidate", "final"],
            "exact_budget_and_classes": True, "speaker_and_prompt_isolation": True,
            "no_replacement_or_byte_duplicates": True, "common_cremad_speakers_and_coverage": True},
            "constructions": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = audit(args.repo.resolve())
    if args.check:
        saved = json.loads(args.check.read_text(encoding="utf-8"))
        require(result == saved, "saved feasibility snapshot does not match current inputs")
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
    print(json.dumps({"pass": True, "checks": result["checks"], "inputs": result["inputs"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
