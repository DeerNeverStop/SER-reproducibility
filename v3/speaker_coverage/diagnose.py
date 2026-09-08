"""Descriptive E0 geometry and historical E1 association; never an E2 gate.

This command does not load new SER predictions, change selection policies, or
fit models. Historical E1 reuses the already scored Study II per-person CSV;
its pooled-confusion UAR is not reconstructed from unavailable raw logits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from v3.data_design.core_plan import digest, file_sha, ranked, read_metadata, require
from v3.speaker_coverage.plan import SCHEMA, load_asv, neutral_label, normalized


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_average(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "invalid rank input")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    start = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    end = np.r_[start[1:], len(values)]
    result = np.empty(len(values), dtype=np.float64)
    for left, right in zip(start, end):
        result[order[left:right]] = (left + right - 1) / 2 + 1
    return result


def pearson(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(left.shape == right.shape and left.ndim == 1, "correlation shape mismatch")
    if len(left) < 3 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    left, right = left - left.mean(), right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator <= 1e-15 else float(np.clip(left @ right / denominator, -1, 1))


def spearman(left, right):
    return pearson(rank_average(left), rank_average(right))


def bootstrap_spearman(left, right, *, repetitions=2000, seed=20260906):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    point = spearman(left, right)
    if point is None:
        return {"rho": None, "ci95": None, "n_people": len(left), "valid_bootstraps": 0,
                "status": "undefined_small_or_constant_sample"}
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(left), size=len(left))
        value = spearman(left[indices], right[indices])
        if value is not None:
            estimates.append(value)
    interval = np.percentile(estimates, [2.5, 97.5], method="linear").tolist() if estimates else None
    return {"rho": point, "ci95": interval, "n_people": len(left),
            "requested_bootstraps": repetitions, "valid_bootstraps": len(estimates),
            "seed": seed, "status": "conditional_descriptive_interval",
            "uncertainty": "resamples_people_after_fixing_models_and_reference_rule; not new-training-population uncertainty"}


def reference_centers(meta, people, prompts, embeddings):
    """Require every specified neutral reference; report, never replace, failures."""
    label = neutral_label(meta)
    centers, support = {}, {}
    dimension = None
    for speaker in sorted(people):
        vectors, paths, missing = [], [], []
        for prompt in prompts:
            row = meta["cells"].get((speaker, prompt, label))
            if row is None:
                missing.append(f"missing_cell:{prompt}")
                continue
            path = row["relative_path"]
            paths.append(path)
            if path not in embeddings:
                missing.append(f"missing_embedding:{path}")
                continue
            try:
                vector = normalized(embeddings[path], path)
                if dimension is None:
                    dimension = len(vector)
                require(len(vector) == dimension, "embedding dimension mismatch")
                vectors.append(vector)
            except ValueError as error:
                missing.append(str(error))
        support[speaker] = {"requested_references": len(prompts), "valid_references": len(vectors),
                            "reference_paths": paths, "problems": missing}
        if len(vectors) == len(prompts) and not missing:
            try:
                centers[speaker] = normalized(np.mean(vectors, axis=0), f"centroid:{speaker}")
            except ValueError as error:
                support[speaker]["problems"].append(str(error))
    return centers, support


def distances(people, centers):
    matrix = np.stack([centers[s] for s in people])
    result = np.clip(1 - matrix @ matrix.T, 0, 2)
    np.fill_diagonal(result, 0)
    return result


def peer_nn3(matrix, index, people):
    peers = [j for j in range(len(people)) if j != index]
    return set(sorted(peers, key=lambda j: (float(matrix[index, j]), people[j]))[:3])


def compare_spaces(first, second, people):
    triangle = np.triu_indices(len(people), 1)
    rows = []
    for i, speaker in enumerate(people):
        peers = [j for j in range(len(people)) if j != i]
        a, b = peer_nn3(first, i, people), peer_nn3(second, i, people)
        rows.append({"speaker": speaker, "distance_rank_spearman": spearman(first[i, peers], second[i, peers]),
                     "nn3_intersection_fraction": len(a & b) / 3,
                     "n_peer_people": len(peers)})
    return rows, {"pairwise_distance_spearman_descriptive": spearman(first[triangle], second[triangle]),
                  "n_candidate_people": len(people), "n_dependent_distance_pairs": len(triangle[0]),
                  "mean_nn3_intersection_fraction": float(np.mean([r["nn3_intersection_fraction"] for r in rows])),
                  "pairs_are_not_independent_samples": True}


def geometry_rows(matrix, people, selected):
    positions = {speaker: i for i, speaker in enumerate(people)}
    selected_indices = [positions[s] for s in selected]
    ordered = np.sort(matrix[:, selected_indices], axis=1)
    global_outlier = matrix.sum(axis=1) / (len(people) - 1)
    percentiles = (rank_average(global_outlier) - .5) / len(people)
    rows = []
    for scope, evaluation in (("including_selected", list(range(len(people)))),
                              ("excluding_selected", [j for j, s in enumerate(people) if s not in set(selected)])):
        require(bool(evaluation), "empty geometry evaluation pool")
        rows.append({"pool_scope": scope, "n_evaluation_people": len(evaluation), "n_candidate_people": len(people),
                     "n_selected_people": len(selected), "nn1_mean": float(ordered[evaluation, 0].mean()),
                     "nn1_max": float(ordered[evaluation, 0].max()),
                     "nn3_mean": float(ordered[evaluation, :3].mean()),
                     "selected_global_loso_distance_mean": float(global_outlier[selected_indices].mean()),
                     "pool_global_loso_distance_mean": float(global_outlier.mean()),
                     "selected_outlier_mean_percentile": float(percentiles[selected_indices].mean())})
    return rows


def e0_diagnostics(meta, plan, asv, xvector=None):
    """Read metadata and selected IDs; never access SER unit output files."""
    by_path = {r["relative_path"]: r for r in meta["cells"].values()}
    contexts = defaultdict(list)
    for unit in plan["units"]:
        if unit["model"] == "ridge_wavlm" and unit["block"] == "core":
            contexts[unit["fold"], unit["rotation"]].append(unit)
    require(len(contexts) == 30, "E0 requires the complete thirty-context plan")
    geometry, stability, overlap, summary, failures = [], [], [], [], []
    for (fold, rotation), units in sorted(contexts.items()):
        require({(u["policy"], u["draw"]) for u in units} == {(s, d) for s in ("U", "R", "C") for d in range(3)},
                "incomplete policy/draw context")
        reference_unit = units[0]
        excluded = {by_path[path]["speaker"] for key in ("test", "val") for path in reference_unit[key]}
        prompts = sorted({by_path[path]["sentence"] for path in reference_unit["fit"]})
        require(len(prompts) == 8, "E0 context does not have eight training prompts")
        candidates = sorted(s for s in meta["speakers"] if s not in excluded and all(
            (s, prompt, label) in meta["cells"] for prompt in prompts for label in range(6)))
        halves = ranked(prompts, f"diagnostic-halves:20260906:{fold}:{rotation}")
        spaces = {}
        for encoder, vectors in (("ecapa", asv), ("xvector", xvector)):
            if vectors is None:
                continue
            full, support = reference_centers(meta, candidates, prompts, vectors)
            first, _ = reference_centers(meta, candidates, halves[:4], vectors)
            second, _ = reference_centers(meta, candidates, halves[4:], vectors)
            missing = sorted(set(candidates) - (set(full) & set(first) & set(second)))
            if missing:
                failure = {"fold": fold, "rotation": rotation, "encoder": encoder,
                           "status": "incomplete_fixed_pool_references", "people": missing,
                           "support": {s: support[s] for s in missing}}
                failures.append(failure)
                require(encoder != "ecapa", "primary ASV does not cover the planned eligible pool")
                continue
            matrix = distances(candidates, full)
            spaces[encoder] = matrix
            rows, aggregate = compare_spaces(distances(candidates, first), distances(candidates, second), candidates)
            summary.append({"fold": fold, "rotation": rotation, "comparison": f"{encoder}_half4_vs_half4",
                            "first_prompts": halves[:4], "second_prompts": halves[4:], **aggregate})
            stability.extend({"fold": fold, "rotation": rotation, "comparison": f"{encoder}_half4_vs_half4", **row}
                             for row in rows)
            for unit in sorted(units, key=lambda u: (u["draw"], u["policy"])):
                selected = sorted({by_path[path]["speaker"] for path in unit["fit"]})
                require(set(selected) <= set(candidates) and len(selected) == 24, "selected identities outside fixed pool")
                geometry.extend({"fold": fold, "rotation": rotation, "draw": unit["draw"],
                                 "policy": unit["policy"], "encoder": encoder, **row}
                                for row in geometry_rows(matrix, candidates, selected))
        if "xvector" in spaces:
            rows, aggregate = compare_spaces(spaces["ecapa"], spaces["xvector"], candidates)
            summary.append({"fold": fold, "rotation": rotation, "comparison": "ecapa_vs_xvector", **aggregate})
            stability.extend({"fold": fold, "rotation": rotation, "comparison": "ecapa_vs_xvector", **row} for row in rows)
        for draw in range(3):
            selected = {u["policy"]: {by_path[path]["speaker"] for path in u["fit"]} for u in units if u["draw"] == draw}
            for a, b in (("U", "R"), ("U", "C"), ("R", "C")):
                overlap.append({"fold": fold, "rotation": rotation, "draw": draw, "policy_a": a, "policy_b": b,
                                "n_intersection": len(selected[a] & selected[b]),
                                "intersection_fraction_of_24": len(selected[a] & selected[b]) / 24,
                                "jaccard": len(selected[a] & selected[b]) / len(selected[a] | selected[b])})
    report = {"status": "completed_descriptive_ecapa_geometry", "n_fold_rotation_contexts": 30,
              "xvector_status": "not_supplied" if xvector is None else ("incomplete" if failures else "completed"),
              "reference_support_failures": failures, "space_summaries": summary,
              "thresholds": None, "policy_or_budget_changes": [], "used_new_ser_scores": False,
              "population": "eligible_training_candidates_only; repeated context rows and distance pairs are dependent",
              "pool_metrics": "cosine nn1 mean/max and nn3 mean; selected people have nn1=0 but generally nn3>0",
              "outlier_measure": "mean distance to other eligible candidates; a descriptive percentile, not a quality exclusion"}
    return report, {"e0_policy_geometry.csv": geometry, "e0_reference_stability.csv": stability,
                    "e0_policy_overlap.csv": overlap}


def legacy_scores(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        if (row["model"], int(row["B"]), int(row["S"]), row["scenario"]) != ("cnn", 576, 48, "prompt_new"):
            continue
        speaker = str(row["speaker"])
        require(speaker not in selected, "duplicate primary legacy person row")
        values = [float(row[f"draw{draw}_uar_pp"]) for draw in range(3)]
        require(np.isfinite(values).all() and all(0 <= value <= 100 for value in values), "invalid legacy UAR")
        require(abs(float(row["mean_uar_pp"]) - np.mean(values)) < 1e-8, "legacy mean does not match three draws")
        selected[speaker] = values
    require(bool(selected), "primary CNN/B576/S48/prompt_new rows absent")
    return selected


def e1_diagnostics(meta, old_plan, scores, asv, *, repetitions=2000):
    """Historical person UAR only; E2 outcomes have no input in this interface."""
    by_path = {r["relative_path"]: r for r in meta["cells"].values()}
    units = [u for u in old_plan["units"] if (u["model"], u["B"], u["S"], u["scenario"]) == ("cnn", 576, 48, "prompt_new")]
    require(len(units) == 90 and len({(u["fold"], u["rotation"], u["draw"]) for u in units}) == 90,
            "incomplete legacy primary 5-fold/6-rotation/3-draw grid")
    references_by_prompts, observations, expected, omissions = {}, defaultdict(dict), defaultdict(set), []
    for unit in units:
        train_people = sorted({by_path[path]["speaker"] for path in unit["fit"]})
        target_people = sorted({by_path[path]["speaker"] for path in unit["test"]})
        prompts = tuple(sorted({by_path[path]["sentence"] for path in unit["fit"]}))
        query_prompts = {by_path[path]["sentence"] for path in unit["test"]}
        require(len(train_people) == 48 and len(unit["fit"]) == 576 and len(prompts) == 8,
                "legacy primary fit budget/people/prompt mismatch")
        require(set(prompts).isdisjoint(query_prompts) and set(train_people).isdisjoint(target_people),
                "legacy reference/query boundary mismatch")
        if prompts not in references_by_prompts:
            references_by_prompts[prompts] = reference_centers(meta, meta["speakers"], prompts, asv)
        centers, support = references_by_prompts[prompts]
        require(set(train_people) <= set(centers), "legacy fit identity has incomplete eight-prompt reference support")
        fit_matrix = np.stack([centers[s] for s in train_people])
        global_people = sorted(centers)
        global_matrix = np.stack([centers[s] for s in global_people])
        for speaker in target_people:
            key = unit["draw"], unit["rotation"]
            require(key not in expected[speaker], "repeated historical person/draw/rotation")
            expected[speaker].add(key)
            if speaker not in centers:
                omissions.append({"speaker": speaker, "draw": key[0], "rotation": key[1],
                                  "fold": unit["fold"], **support[speaker]})
                continue
            nearest = np.sort(np.clip(1 - fit_matrix @ centers[speaker], 0, 2))[:3]
            global_distances = np.clip(1 - global_matrix @ centers[speaker], 0, 2)
            global_distance = float(np.mean([v for person, v in zip(global_people, global_distances) if person != speaker]))
            observations[speaker][key] = {"nn3": float(nearest.mean()), "global_loso": global_distance,
                                          "reference_count": 8, "global_loso_people": len(global_people) - 1}
    required = {(d, r) for d in range(3) for r in range(6)}
    require(all(keys == required for keys in expected.values()), "legacy query coverage not six rotations per draw")
    require(set(expected) <= set(scores), "legacy CSV missing expected primary people")
    people, excluded = [], []
    for speaker in sorted(expected):
        if set(observations[speaker]) != required:
            excluded.append({"speaker": speaker, "reason": "incomplete_eight_reference_support_in_one_or_more_contexts",
                             "valid_contexts": len(observations[speaker]), "expected_contexts": 18})
            continue
        row = {"speaker": speaker, "mean_uar_pp": float(np.mean(scores[speaker])),
               "nn3_distance": float(np.mean([observations[speaker][key]["nn3"] for key in sorted(required)])),
               "global_loso_mean_distance": float(np.mean([observations[speaker][key]["global_loso"] for key in sorted(required)])),
               "n_contexts": 18, "references_per_context": 8}
        row.update({f"draw{draw}_uar_pp": scores[speaker][draw] for draw in range(3)})
        people.append(row)
    require(len(people) >= 3, "too few complete-reference people for E1")
    x = np.array([r["nn3_distance"] for r in people])
    y = np.array([r["mean_uar_pp"] for r in people])
    g = np.array([r["global_loso_mean_distance"] for r in people])
    covariates = np.column_stack([np.ones(len(g)), rank_average(g)])
    residual_x = rank_average(x) - covariates @ np.linalg.lstsq(covariates, rank_average(x), rcond=None)[0]
    residual_y = rank_average(y) - covariates @ np.linalg.lstsq(covariates, rank_average(y), rcond=None)[0]
    partial = (pearson(residual_x, residual_y)
               if min(np.linalg.norm(residual_x), np.linalg.norm(residual_y)) > 1e-10 else None)
    report = {"status": "completed_historical_csv_association", "primary": "CNN/B576/S48/prompt_new; nearest3 versus person UAR",
              "n_legacy_people": len(expected), "n_complete_reference_people": len(people),
              "n_legacy_fit_instances": 90, "n_reference_contexts_per_person": 18,
              "primary_spearman": bootstrap_spearman(x, y, repetitions=repetitions),
              "global_loso_spearman": bootstrap_spearman(g, y, repetitions=repetitions),
              "rank_partial_correlation_controlling_global_loso_descriptive": partial,
              "global_loso_definition": "mean cosine distance to every other corpus person with complete same-eight-prompt neutral references; never selected by outcomes",
              "uar_aggregation": "existing core_score.collect_draws pools six rotations' class counts, computes fixed-six-class UAR, then averages three draw UARs; raw logits not replayed here",
              "distance_aggregation": "equal mean over six rotations within draw, then three draws",
              "reference_query_scope": "disjoint within each historical unit; a reference may be a query in another rotation",
              "excluded_people": excluded, "reference_support_failures": omissions,
              "voice_only_covariate": {"status": "not_supplied_or_aligned", "included_in_model": False},
              "causal_interpretation": False, "used_new_ser_scores": False}
    return report, people


def resolve_legacy(repo, study2_root=None, explicit_csv=None, explicit_plan=None):
    root = Path(study2_root) if study2_root else Path(repo)
    if explicit_csv:
        csv_path = Path(explicit_csv)
    else:
        candidates = [root / "v3/data_design/results/per_speaker.csv", root / "paper/study2/results/per_speaker.csv"]
        available = [path for path in candidates if path.is_file()]
        require(bool(available), "historical per_speaker.csv not found")
        require(len({file_sha(path) for path in available}) == 1, "ambiguous historical CSV copies; supply --legacy-per-speaker")
        csv_path = available[0]
    plan_path = Path(explicit_plan) if explicit_plan else root / "v3/data_design/evidence/core_plan.json.gz"
    require(csv_path.is_file() and plan_path.is_file(), "historical CSV/plan missing")
    return csv_path, plan_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--asv", type=Path, required=True)
    parser.add_argument("--xvector", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-per-speaker", type=Path)
    parser.add_argument("--study2-root", type=Path)
    parser.add_argument("--legacy-plan", type=Path)
    args = parser.parse_args()
    require(args.repo.resolve() == Path(__file__).resolve().parents[2], 'executed checkout differs from --repo')
    plan = read_json(args.plan)
    require(plan["schema"] == SCHEMA and plan["plan_sha256"] == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "new plan schema/hash mismatch")
    require(file_sha(args.asv) == plan["input"]["asv_sha256"], "primary ASV differs from frozen selection input")
    require(plan['source_sha256'].get('v3/speaker_coverage/diagnose.py') == file_sha(__file__), 'diagnostic source differs from frozen plan')
    meta = read_metadata(args.repo.resolve())
    require(all(meta['input'][k] == plan['input'][k] for k in ('manifest_path','manifest_sha256','demographics_path','demographics_sha256')), 'metadata differs from frozen plan')
    asv, asv_identity = load_asv(args.asv)
    xvector, xvector_identity = load_asv(args.xvector) if args.xvector else (None, None)
    e0, tables = e0_diagnostics(meta, plan, asv, xvector)
    report = {"schema": "ser-speaker-coverage-diagnostics-1", "plan_sha256": plan["plan_sha256"],
              "input": {"asv": asv_identity, "xvector": xvector_identity,
                        "diagnose_source_sha256": file_sha(Path(__file__)),
                        "legacy_aggregation_source_sha256": file_sha(args.repo / "v3/data_design/core_score.py")},
              "descriptive_only": True, "changes_to_policy_or_budget": [], "new_ser_scores_read": False, "E0": e0,
              "E1": {"status": "not_requested; provide --legacy-per-speaker or --study2-root"}}
    if args.legacy_per_speaker or args.study2_root:
        csv_path, old_plan_path = resolve_legacy(args.repo, args.study2_root, args.legacy_per_speaker, args.legacy_plan)
        old_plan = read_json(old_plan_path)
        require(old_plan.get("schema") == "ser-study2-core-1"
                and old_plan["plan_sha256"] == digest({k: v for k, v in old_plan.items() if k != "plan_sha256"}),
                "historical plan schema/hash mismatch")
        report["E1"], tables["e1_people.csv"] = e1_diagnostics(meta, old_plan, legacy_scores(csv_path), asv)
        report["E1"]["input"] = {"per_speaker_file": csv_path.name, "per_speaker_sha256": file_sha(csv_path),
                                   "legacy_plan_sha256": old_plan["plan_sha256"], "legacy_plan_file_sha256": file_sha(old_plan_path)}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.out / name, rows)
    report["table_sha256"] = {name: file_sha(args.out / name) for name in tables}
    write_json(args.out / "diagnostics.json", report)
    print(json.dumps({"E0": e0["status"], "E1": report["E1"]["status"],
                      "new_ser_scores_read": False, "policy_changes": []}))


if __name__ == "__main__":
    main()
