"""Outcome-blind speaker-coverage plans; no SER scores or GPU operations.

ASV input is an NPZ with ``paths`` (relative audio names) and ``embeddings``
(N,D). Only eligible training speakers' eight neutral representatives are
looked up in each fold/rotation. Test/stop vectors never enter a selector.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path, PurePosixPath

import numpy as np

from v3.data_design.core_plan import (
    canonical, digest, feature_identity, file_sha, prompt_slots, ranked,
    read_metadata, require, roles, rows_for, write_json,
)


SCHEMA = "ser-speaker-coverage-1"
PROGRAM = "SER26-SPEAKER-COVERAGE-1"
MASTER_SEED = 20260906
POLICIES = ("U", "R", "C")
SEXES = ("Female", "Male")
CORE_MODELS = ("ridge_wavlm", "cnn")
SWAP_TOLERANCE = 1e-12
WAVLM_WEIGHTS_SHA256 = "d8e745fb761c56d209431413bcc373f92784dc2c8d850199b6caa5cf05ca2c36"
BASE_SOURCE_PATHS = (
    "v3/speaker_coverage/EXECUTION_PLAN.md",
    "v3/data_design/core_plan.py", "v3/data_design/core_run.py", "v3/data_design/core_verify.py",
    "v3/deploy/engines_deploy.py", "v2/plan_rc2/unit_configs.json",
    "v2/ser_v2/train.py", "v2/ser_v2/corpora.py", "v2/ser_v2/features.py",
    "v2/ser_v2/common.py", "v2/ser_v2/stats.py", "advanced_models.py",
    "tuned_standard_experiment.py", "advanced_experiment_utils.py",
    "experiment_utils.py", "features.py", "dataset.py", "fno_model.py",
)


def seed_for(*parts):
    return int(digest([MASTER_SEED, *parts])[:16], 16) % (2**31 - 1)


def safe_audio_path(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    require(isinstance(value, str) and bool(value), "ASV path must be a nonempty string")
    p = PurePosixPath(value)
    require("\\" not in value and ":" not in value and not p.is_absolute()
            and ".." not in p.parts and str(p) == value, "unsafe/noncanonical ASV path")
    return value


def load_asv(path):
    """Load input without pickle; numerical eligibility is checked on lookup."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        require({"paths", "embeddings"}.issubset(archive.files), "ASV keys paths/embeddings required")
        paths = archive["paths"]
        vectors = archive["embeddings"]
        require(paths.ndim == 1 and paths.dtype.kind in "US", "ASV paths must be string vector")
        require(vectors.ndim == 2 and vectors.shape[0] == len(paths)
                and vectors.shape[1] > 0 and vectors.dtype.kind == "f", "invalid ASV matrix")
        names = [safe_audio_path(p.item()) for p in paths]
        require(len(set(names)) == len(names), "duplicate ASV paths")
        # Copy rows out of the archive, but do not derive any full-population
        # statistic or scientific gate from stop/query embeddings.
        mapping = {name: np.array(vectors[i], dtype=np.float64, copy=True)
                   for i, name in enumerate(names)}
    return mapping, {"asv_path": path.name, "asv_location": "features",
                     "asv_sha256": file_sha(path), "asv_rows": len(names),
                     "asv_dimension": int(vectors.shape[1])}


def normalized(vector, context):
    value = np.asarray(vector, dtype=np.float64)
    require(value.ndim == 1 and value.size > 0 and np.isfinite(value).all(),
            f"invalid ASV vector: {context}")
    norm = float(np.linalg.norm(value))
    require(np.isfinite(norm) and norm > 1e-12, f"zero ASV norm: {context}")
    return value / norm


def neutral_label(meta):
    labels = {int(row["label_index"]) for row in meta["cells"].values()
              if str(row.get("label", "")).lower() in ("neutral", "neu")}
    require(len(labels) == 1, "unique neutral label index required")
    return labels.pop()


def candidate_centroids(meta, candidates, prompts, embeddings):
    """Only read the supplied training candidates' neutral reference cells."""
    label = neutral_label(meta)
    centers, references = {}, {}
    dimension = None
    for speaker in sorted(candidates):
        paths = [meta["cells"][(speaker, prompt, label)]["relative_path"]
                 for prompt in sorted(prompts)]
        require(len(paths) == len(set(paths)) == 8, "eight independent neutral representatives required")
        vectors = []
        for path in paths:
            require(path in embeddings, f"missing training reference embedding: {path}")
            vector = normalized(embeddings[path], path)
            if dimension is None:
                dimension = vector.size
            require(vector.size == dimension, "inconsistent ASV embedding dimensions")
            vectors.append(vector)
        centers[speaker] = normalized(np.mean(vectors, axis=0), f"centroid:{speaker}")
        references[speaker] = paths
    return centers, references


def _distance_matrix(candidates, centers):
    matrix = np.stack([normalized(centers[s], f"centroid:{s}") for s in candidates])
    distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def pool_metrics(distances, selected):
    nearest = np.min(distances[:, selected], axis=1)
    return {"mean_nn1": float(np.mean(nearest)), "max_nn1": float(np.max(nearest))}


def select_policy(candidates, centers, sex, policy, fold, rotation, draw=0):
    """Select from one fixed eligible pool; all objectives are pool-level nn1."""
    candidates = sorted(candidates)
    require(policy in POLICIES, "unknown selection policy")
    require(len(candidates) == len(set(candidates)), "duplicate candidate")
    require(all(sex[s] in SEXES for s in candidates), "invalid sex stratum")
    require(all(sum(sex[s] == group for s in candidates) >= 12 for group in SEXES),
            "fewer than twelve complete candidates in a sex stratum")
    distances = _distance_matrix(candidates, centers)
    tie_salt = f"coverage:{MASTER_SEED}:{fold}:{rotation}:{policy}"
    ordered = ranked(candidates, tie_salt)
    tie_rank = {s: i for i, s in enumerate(ordered)}
    index = {s: i for i, s in enumerate(candidates)}
    selected, trace, counts = [], [], Counter()
    if policy == "U":
        for group in SEXES:
            eligible = [s for s in candidates if sex[s] == group]
            rng = np.random.Generator(np.random.PCG64(seed_for("U", fold, rotation, draw, group)))
            selected.extend(index[s] for s in rng.choice(eligible, size=12, replace=False).tolist())
        termination = "uniform_quota_complete"
    else:
        first = index[ordered[0]]
        selected.append(first)
        counts[sex[candidates[first]]] += 1
        trace.append({"step": "initialize", "speaker": candidates[first],
                      **pool_metrics(distances, selected)})
        while len(selected) < 24:
            available = [j for j, s in enumerate(candidates)
                         if j not in selected and counts[sex[s]] < 12]
            require(bool(available), "selection exhausted before quota completion")
            nearest = np.min(distances[:, selected], axis=1)
            if policy == "R":
                key = lambda j: (float(np.minimum(nearest, distances[:, j]).mean()), tie_rank[candidates[j]])
            else:
                key = lambda j: (-float(nearest[j]), tie_rank[candidates[j]])
            chosen = min(available, key=key)
            selected.append(chosen)
            counts[sex[candidates[chosen]]] += 1
            trace.append({"step": "add", "speaker": candidates[chosen],
                          **pool_metrics(distances, selected)})
        termination = "quota_complete"
        if policy == "R":
            termination = "no_strict_swap_improvement"
            for swap_number in range(100):
                current = pool_metrics(distances, selected)["mean_nn1"]
                best = None
                unselected = [j for j in range(len(candidates)) if j not in selected]
                for outgoing in selected:
                    retained = [j for j in selected if j != outgoing]
                    nearest = np.min(distances[:, retained], axis=1)
                    for incoming in unselected:
                        if sex[candidates[incoming]] != sex[candidates[outgoing]]:
                            continue
                        score = float(np.minimum(nearest, distances[:, incoming]).mean())
                        if score < current - SWAP_TOLERANCE:
                            proposal = (score, tie_rank[candidates[outgoing]],
                                        tie_rank[candidates[incoming]], outgoing, incoming)
                            if best is None or proposal < best:
                                best = proposal
                if best is None:
                    break
                outgoing, incoming = best[-2:]
                selected[selected.index(outgoing)] = incoming
                trace.append({"step": "same_sex_swap", "swap": swap_number + 1,
                              "out": candidates[outgoing], "in": candidates[incoming],
                              **pool_metrics(distances, selected)})
            else:
                termination = "swap_limit_100"
    people = sorted(candidates[j] for j in selected)
    require(len(people) == len(set(people)) == 24, "speaker count mismatch")
    require(Counter(sex[s] for s in people) == Counter({"Female": 12, "Male": 12}), "sex quota mismatch")
    return people, {"policy": policy, "selected_speakers": people,
                    "evaluation_population": "same_complete_training_candidate_pool_including_selected",
                    "distance_definition": "cosine_nn1", "termination": termination,
                    "selection_trace": trace, **pool_metrics(distances, selected)}


def abstract_layout(fold, rotation, draw, attempt=0):
    """A common slot graph, independent of policy and selected identities."""
    require(0 <= attempt < 100, "layout attempt out of range")
    subsets = list(combinations(range(8), 4))
    graph = {}
    for group in SEXES:
        assignments = []
        for pair in range(6):
            salt = f"layout:{MASTER_SEED}:{fold}:{rotation}:{draw}:{attempt}:{group}:{pair}"
            first = ranked(subsets, salt)[0]
            second = tuple(i for i in range(8) if i not in first)
            assignments.extend([list(first), list(second)])
        graph[group] = assignments
    return graph


def make_panel(meta, selected, prompts, graph, fold, rotation):
    require(len(prompts) == len(set(prompts)) == 8, "exactly eight training prompts required")
    require(len(selected) == len(set(selected)) == 24, "exactly 24 distinct selected people required")
    paths, assignment = [], {}
    for group in SEXES:
        people = ranked([s for s in selected if meta["sex"][s] == group],
                        f"layout-mapping:{MASTER_SEED}:{fold}:{rotation}:{group}")
        require(len(people) == 12 and len(graph[group]) == 12, "layout sex cardinality mismatch")
        for speaker, slots in zip(people, graph[group]):
            require(len(slots) == len(set(slots)) == 4 and set(slots) <= set(range(8)),
                    "invalid four-prompt allocation")
            assignment[speaker] = [prompts[i] for i in slots]
            for prompt in assignment[speaker]:
                for label in range(6):
                    require((speaker, prompt, label) in meta["cells"], "selected incomplete prompt/class cell")
                    paths.append(meta["cells"][(speaker, prompt, label)]["relative_path"])
    paths.sort()
    validate_panel(meta, paths)
    return paths, assignment


def validate_panel(meta, paths):
    by_path = {row["relative_path"]: row for row in meta["cells"].values()}
    require(len(paths) == len(set(paths)) == 576, "fit requires 576 distinct paths")
    rows = [by_path[p] for p in paths]
    require(len({r["sha256"] for r in rows}) == 576, "fit contains byte duplicates")
    people = {r["speaker"] for r in rows}
    require(len(people) == 24, "fit requires 24 people")
    require(Counter(meta["sex"][s] for s in people) == Counter({"Female": 12, "Male": 12}), "fit sex quota")
    require(set(Counter(r["speaker"] for r in rows).values()) == {24}, "per-person recording quota")
    prompt_counts = Counter(r["sentence"] for r in rows)
    require(len(prompt_counts) == 8 and set(prompt_counts.values()) == {72}, "prompt recording quota")
    sex_prompt = Counter((meta["sex"][r["speaker"]], r["sentence"]) for r in rows)
    require(len(sex_prompt) == 16 and set(sex_prompt.values()) == {36}, "sex-by-prompt quota")
    cells = Counter((r["speaker"], r["sentence"], int(r["label_index"])) for r in rows)
    require(set(cells.values()) == {1}, "repeated speaker/prompt/class")
    blocks = Counter((r["speaker"], r["sentence"]) for r in rows)
    require(set(blocks.values()) == {6}, "incomplete six-class block")


def _check_boundaries(meta, fit, val, test):
    rows = {r["relative_path"]: r for r in meta["cells"].values()}
    sets = [set(part) for part in (fit, val, test)]
    require(all(len(part) == len(set(part)) for part in (fit, val, test)), "duplicate role path")
    for a, b in combinations(range(3), 2):
        require(sets[a].isdisjoint(sets[b]), "fit/val/test path overlap")
        require({rows[p]["speaker"] for p in sets[a]}.isdisjoint({rows[p]["speaker"] for p in sets[b]}),
                "fit/val/test speaker overlap")
        require({rows[p]["sha256"] for p in sets[a]}.isdisjoint({rows[p]["sha256"] for p in sets[b]}),
                "fit/val/test byte overlap")


def plan_context(meta, embeddings, fold, rotation):
    """Create all policy/draw panels for one outer fold and query pair."""
    test_people, stop_people, pool = roles(meta, 0, fold)
    query_prompts, stop_prompts, slots = prompt_slots(meta, rotation)
    train_prompts = slots["prompt_new"]
    require(len(test_people) > 0 and len(stop_people) == 8, "outer/stop role capacity")
    require(set(train_prompts).isdisjoint(query_prompts) and set(stop_prompts) <= set(train_prompts),
            "prompt role overlap")
    eligible = sorted(s for s in pool if all((s, p, c) in meta["cells"]
                                           for p in train_prompts for c in range(6)))
    require(all(sum(meta["sex"][s] == group for s in eligible) >= 12 for group in SEXES),
            "inadequate complete eight-prompt candidate capacity")
    centers, refs = candidate_centroids(meta, eligible, train_prompts, embeddings)
    val = [r["relative_path"] for r in rows_for(meta, stop_people, stop_prompts)]
    test = [r["relative_path"] for r in rows_for(meta, test_people, query_prompts)]
    chosen, selections = {}, []
    for policy in POLICIES:
        for draw in (range(3) if policy == "U" else (0,)):
            people, diagnostic = select_policy(eligible, centers, meta["sex"], policy, fold, rotation, draw)
            diagnostic.update(fold=fold, rotation=rotation, draw=draw if policy == "U" else None)
            selections.append(diagnostic)
            for d in (range(3) if policy != "U" else (draw,)):
                chosen[policy, d] = people
    panels, layouts = [], []
    previous = {policy: set() for policy in POLICIES}
    for draw in range(3):
        for attempt in range(100):
            graph = abstract_layout(fold, rotation, draw, attempt)
            trial = {policy: make_panel(meta, chosen[policy, draw], train_prompts, graph, fold, rotation)
                     for policy in POLICIES}
            identities = {policy: digest(trial[policy][0]) for policy in POLICIES}
            if all(identities[p] not in previous[p] for p in POLICIES):
                break
        else:
            raise ValueError("no distinct common layout within 100 predefined attempts")
        layouts.append({"draw": draw, "attempt": attempt, "abstract_slots": graph})
        for policy in POLICIES:
            fit, assignment = trial[policy]
            previous[policy].add(identities[policy])
            _check_boundaries(meta, fit, val, test)
            panels.append({"policy": policy, "fold": fold, "rotation": rotation, "draw": draw,
                           "fit": fit, "val": val, "test": test,
                           "fit_manifest_sha256": identities[policy],
                           "layout_attempt": attempt, "speaker_prompts": assignment})
    query_counts = Counter((r["speaker"], int(r["label_index"]))
                           for r in rows_for(meta, test_people, query_prompts))
    capacity = {"fold": fold, "rotation": rotation, "eligible": len(eligible),
                "eligible_female": sum(meta["sex"][s] == "Female" for s in eligible),
                "eligible_male": sum(meta["sex"][s] == "Male" for s in eligible),
                "train_pool": len(pool), "test_speakers": sorted(test_people),
                "stop_speakers": sorted(stop_people), "eligible_speakers": eligible,
                "train_prompts": train_prompts, "stop_prompts": stop_prompts,
                "query_prompts": query_prompts, "test_rows": len(test), "val_rows": len(val),
                "minimum_query_per_person_class": min(query_counts.values()),
                "reference_paths": refs, "reference_sha256": digest(refs),
                "centroid_sha256": digest({s: centers[s].tolist() for s in eligible})}
    return panels, capacity, {"fold": fold, "rotation": rotation, "selections": selections,
                              "layouts": layouts}


def model_config(model):
    if model == "ridge_wavlm":
        return {"model": "ridge_a1_wavlm_base_plus", "feature_state": 12,
                "alpha": 1.0, "class_weight": "balanced", "solver": "lsqr", "tol": 0.0001}
    if model == "cnn":
        return {"model": "cnn", "engine": "p1_frozen", "batch_size": 32, "epochs": 100,
                "patience": 101, "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1,
                "early_stopping": False, "checkpoint_selection": "min_val_loss"}
    require(model == "wavlm_ft", "unknown model")
    return {"model": "wavlm_base_plus_ft", "engine": "wavlm_partial_ft", "batch_size": 16,
            "crop_seconds": 3.0, "eval_cap_seconds": 10.0, "epochs": 15, "patience": 16,
            "fp16": True, "lr_encoder": 5e-5, "lr_head": 0.001, "weight_decay": 0.01,
            "trainable_layers": "top4+pool+head", "early_stopping": False,
            "checkpoint_selection": "min_val_loss"}


def _unit(panel, model, seed_index=0):
    block = "ft" if model == "wavlm_ft" else "core"
    seed = seed_for("training", block, panel["fold"], panel["rotation"], panel["draw"], seed_index)
    unit = {k: panel[k] for k in ("policy", "fold", "rotation", "draw", "fit", "val", "test")}
    unit.update(block=block, model=model, seed_index=seed_index, train_seed=seed,
                B=576, S=24, P_global=8, P_per_speaker=4, R=1, scenario="prompt_new",
                config=model_config(model), fit_manifest_sha256=panel["fit_manifest_sha256"])
    unit["unit_id"] = digest(unit)
    return unit


def build_from_metadata(meta, embeddings, inputs, *, source_sha256=None, runtime_ready=False):
    """Pure construction entry point; synthetic callers must label their inputs."""
    require(len(meta["prompts"]) == 12, "exactly twelve prompt families required")
    require(len(meta["speakers"]) == len(set(meta["speakers"])), "duplicate metadata speaker")
    units, capacities, diagnostics = [], [], []
    for fold in range(5):
        for rotation in range(6):
            panels, capacity, diagnostic = plan_context(meta, embeddings, fold, rotation)
            capacities.append(capacity)
            diagnostics.append(diagnostic)
            for panel in panels:
                units.extend(_unit(panel, model) for model in CORE_MODELS)
                if panel["draw"] == 0:
                    units.extend(_unit(panel, "wavlm_ft", seed_index) for seed_index in range(2))
    counts = dict(Counter(u["model"] for u in units))
    require(counts == {"ridge_wavlm": 270, "cnn": 270, "wavlm_ft": 180}, "complete block cardinality")
    require(len({u["unit_id"] for u in units}) == len(units) == 720, "duplicate unit identities")
    design = {"status": "prospectively_specified_estimation_historically_exposed_corpus",
              "prior_exposure": "Existing CREMA-D results and Claude review informed design.",
              "master_seed": MASTER_SEED, "outer_roles": "core_plan.roles(meta,0,fold); fixed across draw",
              "speaker_folds": 5, "prompt_rotations": 6, "draws": 3,
              "policies": list(POLICIES), "blocks": {"core": 540, "ft": 180}, "expected_units": 720,
              "ft_draws": [0], "ft_train_seeds": 2,
              "pilot_units": {"core": 18, "ft": 1, "included_in_formal_units": False},
              "B": 576, "S": 24, "stop_speakers": 8, "reference": "eight_train_prompt_neutral_representatives",
              "centroid": "normalize_each_then_mean_then_normalize_float64",
              "selection_metric": "cosine_nn1_on_same_eligible_training_pool_including_selected",
              "selection_queries": "none; no test/stop embeddings or scores used",
              "R_swap_limit": 100, "R_strict_improvement_tolerance": SWAP_TOLERANCE,
              "layout_attempt_limit": 100,
              "primary_contrasts": {"primary": {"model": "cnn", "contrast": "C-U",
                                                  "endpoint": "person_mean_uar_pp", "inference": "two_sided_estimation"},
                                    "required_second_model": {"model": "wavlm_ft", "contrast": "C-U"},
                                    "secondary": "R-U; Ridge contrasts; other fixed contrasts",
                                    "q25": "descriptive_distribution_quantile_difference",
                                    "sign_dependent_gate": False},
              "scientific_scores_hidden_until_complete": True,
              "runtime_ready": bool(runtime_ready), "sourcefreeze_complete": bool(runtime_ready),
              "ft_runtime_ready": bool(runtime_ready and inputs.get("wavlm_model_sha256")),
              "fixture": bool(inputs.get("fixture", False))}
    plan = {"schema": SCHEMA, "program": PROGRAM, "input": dict(inputs),
            "source_sha256": dict(source_sha256 or {}), "design": design, "units": units}
    plan["plan_sha256"] = digest(plan)
    capacity = {"schema": "ser-speaker-coverage-capacity-1", "plan_sha256": plan["plan_sha256"],
                "representative_cells": len(meta["cells"]), "people": len(meta["speakers"]),
                "by_model": counts, "conditions": capacities, "fixture": design["fixture"]}
    diagnostics = {"schema": "ser-speaker-coverage-selection-1", "plan_sha256": plan["plan_sha256"],
                   "metric": "pool_in_sample_cosine_nn1", "conditions": diagnostics,
                   "no_heldout_distance_gate": True, "fixture": design["fixture"]}
    return plan, capacity, diagnostics


def build(repo, features, asv_npz, *, wavlm_model=None, source_paths=None, require_runtime=True):
    repo, features, asv_npz = Path(repo).resolve(), Path(features).resolve(), Path(asv_npz).resolve()
    meta = read_metadata(repo)
    embeddings, asv_input = load_asv(asv_npz)
    receipt = asv_npz.with_suffix(".json")
    require(receipt.is_file(), "ASV extraction receipt missing")
    inputs = {**meta["input"], **feature_identity(features), **asv_input,
              "asv_receipt_file": receipt.name, "asv_receipt_sha256": file_sha(receipt),
              "wavlm_weights_sha256": WAVLM_WEIGHTS_SHA256}
    require(not inputs.get("fixture"), "fixture cannot enter corpus build")
    if wavlm_model is not None:
        wavlm_model = Path(wavlm_model)
        require(wavlm_model.is_file(), "WavLM model file missing")
        inputs.update(wavlm_model_file=wavlm_model.name, wavlm_model_sha256=file_sha(wavlm_model))
    package_sources = [p.relative_to(repo).as_posix() for p in (repo / "v3/speaker_coverage").glob("*.py")]
    sources = sorted(set(source_paths if source_paths is not None else (*BASE_SOURCE_PATHS, *package_sources)))
    require(all((repo / p).is_file() for p in sources), "source pin missing")
    ready = {"v3/speaker_coverage/run.py", "v3/speaker_coverage/plan.py",
             "v3/speaker_coverage/verify.py"} <= set(sources)
    require(not require_runtime or ready, "runtime sources not frozen; use --draft only for inspection")
    pins = {p: file_sha(repo / p) for p in sources}
    return build_from_metadata(meta, embeddings, inputs, source_sha256=pins, runtime_ready=ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--asv", type=Path, required=True)
    parser.add_argument("--wavlm-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    plan, capacity, diagnostics = build(args.repo, args.features, args.asv,
                                         wavlm_model=args.wavlm_model, require_runtime=not args.draft)
    write_json(args.out / "plan.json", plan)
    write_json(args.out / "capacity.json", capacity)
    write_json(args.out / "selection_diagnostics.json", diagnostics)
    print(canonical({"pass": True, "plan_sha256": plan["plan_sha256"], "units": len(plan["units"]),
                     "minimum_eligible": min(c["eligible"] for c in capacity["conditions"]),
                     "runtime_ready": plan["design"]["runtime_ready"]}))


if __name__ == "__main__":
    main()
