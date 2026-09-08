"""Fallible voice-only human votes as a descriptive historical E1 covariate.

No new SER outputs, selection decisions, model fitting, or inferential gate.
The raw vote table's first column encodes 100000 * queryType + clipNum;
official corpus code identifies queryType 1 as voice-only. Audio stimuli were
MP3s, whereas the computational manifest describes WAVs. Known bad stimuli
are excluded explicitly, not treated as clean human ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from v3.data_design.core_plan import digest, file_sha, read_metadata, require

OFFICIAL = "https://github.com/CheyneyComputerScience/CREMA-D"
EMOTIONS = "ADF HNS".replace(" ", "")
LABELS = {"angry": "A", "disgust": "D", "fearful": "F", "happy": "H", "neutral": "N", "sad": "S"}
KNOWN_STIMULUS_ISSUES = {
    "1076_MTI_NEU_XX": "MP3 very short/no audio; computational WAV is fine",
    "1076_MTI_SAD_XX": "MP3 and WAV no audio/very short; also removed by corpus hygiene",
    "1064_TIE_SAD_XX": "MP3 has no duration",
    "1064_IEO_DIS_MD": "MP3 contains about one minute of multiple emotional displays",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "invalid CSV header")
        return list(reader)


def read_json(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def audit_votes(corpus_root):
    root = Path(corpus_root)
    source_names = ("README.md", "tabulateVotesV2.r", "summarizeVotes.r", "SentenceFilenames.csv",
                    "processedResults/tabulatedVotes.csv", "processedResults/summaryTable.csv")
    sources = {name: file_sha(root / name) for name in source_names}
    tabulate = (root / "tabulateVotesV2.r").read_text(encoding="utf-8")
    summarize = (root / "summarizeVotes.r").read_text(encoding="utf-8")
    require("(100000*testResponses$queryType) + testResponses$clipNum" in tabulate,
            "source code does not establish queryType/clip encoding")
    require("voiceTestResponses <- subset(testResponses,queryType==1)" in summarize,
            "source code does not identify voice-only queryType")
    names = {int(r["Stimulus_Number"]): r["Filename"] for r in read_csv(root / "SentenceFilenames.csv")}
    require(len(names) == 7442 and set(names) == set(range(1, 7443)), "unexpected stimulus lookup population")
    summaries = read_csv(root / "processedResults/summaryTable.csv")
    require(len(summaries) == 7442 and len({r["FileName"] for r in summaries}) == 7442, "invalid summary population")
    voice_summary = {r["FileName"]: r["VoiceVote"] for r in summaries}
    rows = read_csv(root / "processedResults/tabulatedVotes.csv")
    require(len(rows) == 22326 and len({r[""] for r in rows}) == 22326, "invalid vote table population/row identity")
    blocks, voices, ties = Counter(), {}, 0
    for row in rows:
        query_type, clip = divmod(int(row[""]), 100000)
        require(query_type in (1, 2, 3) and names.get(clip) == row["fileName"], "vote row encoding/filename disagreement")
        blocks[query_type] += 1
        if query_type != 1:
            continue  # Do not read visual/multimodal emotion votes as a covariate.
        counts = {e: int(row[e]) for e in EMOTIONS}
        n = int(row["numResponses"])
        require(n > 0 and min(counts.values()) >= 0 and sum(counts.values()) == n, "invalid voice counts")
        maximum = max(counts.values())
        winners = tuple(e for e in EMOTIONS if counts[e] == maximum)
        require(row["emoVote"] == ":".join(winners) == voice_summary.get(row["fileName"]), "voice majority disagreement")
        require(abs(float(row["agreement"]) - maximum / n) < 1e-12, "voice agreement/count disagreement")
        require(row["fileName"] not in voices, "duplicate voice-only filename")
        voices[row["fileName"]] = {"counts": counts, "num_responses": n, "winners": winners,
                                  "agreement": maximum / n, "row_id": int(row[""])}
        ties += len(winners) > 1
    require(blocks == Counter({1: 7442, 2: 7442, 3: 7442}), "incomplete modality blocks")
    return voices, {"status": "voice_only_schema_and_counts_verified", "official_repository": OFFICIAL,
                    "sources_sha256": sources, "query_type_used": 1, "encoding": "100000*queryType+clipNum",
                    "source_proof": ["tabulateVotesV2.r: goodQueryTypeClip", "summarizeVotes.r: voiceTestResponses queryType==1",
                                     "README.md: queryType and MP3/AudioWAV descriptions"],
                    "modality_row_counts": dict(blocks), "voice_rows": len(voices), "voice_tied_pluralities": ties,
                    "cross_checks": ["clipNum to SentenceFilenames", "voice counts to numResponses", "max count to agreement",
                                     "all tied modes to emoVote and summary VoiceVote"],
                    "visual_vote_values_used": False, "known_mp3_stimulus_issues": KNOWN_STIMULUS_ISSUES,
                    "stimulus_boundary": "human ratings used original MP3s; computation uses WAVs; byte identity is not asserted"}


def historical_queries(meta, old_plan):
    require(old_plan.get("schema") == "ser-study2-core-1" and old_plan.get("plan_sha256") ==
            digest({k: v for k, v in old_plan.items() if k != "plan_sha256"}), "legacy plan hash/schema mismatch")
    require(meta["input"]["manifest_sha256"] == old_plan["input"]["manifest_sha256"], "legacy/main manifest mismatch")
    by_path = {row["relative_path"]: row for row in meta["cells"].values()}
    units = [u for u in old_plan["units"] if (u["model"], u["B"], u["S"], u["scenario"]) == ("cnn", 576, 48, "prompt_new")]
    require(len(units) == 90 and {(u["fold"], u["rotation"], u["draw"]) for u in units} ==
            {(f, r, d) for f in range(5) for r in range(6) for d in range(3)}, "incomplete historical E1 grid")
    queries, contexts = defaultdict(lambda: defaultdict(set)), defaultdict(set)
    for unit in units:
        require(len(set(unit["test"])) == len(unit["test"]) and set(unit["test"]) <= by_path.keys(), "unknown/repeated query path")
        by_person = defaultdict(list)
        for path in unit["test"]:
            by_person[by_path[path]["speaker"]].append(path)
        for person, paths in by_person.items():
            key = unit["draw"], unit["rotation"]
            require(key not in contexts[person], "duplicate historical person context")
            contexts[person].add(key)
            require(not queries[person][unit["draw"]].intersection(paths), "query repeated across rotations in one draw")
            queries[person][unit["draw"]].update(paths)
    require(all(v == {(d, r) for d in range(3) for r in range(6)} for v in contexts.values()), "incomplete historical person coverage")
    return queries, by_path


def ranked(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(), "nonfinite rank data")
    return np.array([1 + np.count_nonzero(values < v) + (np.count_nonzero(values == v) - 1) / 2 for v in values])


def correlation(x, y, controls=()):
    x, y = ranked(x), ranked(y)
    z = np.column_stack([np.ones(len(x))] + [ranked(c) for c in controls])
    if len(x) <= z.shape[1] + 1 or np.linalg.matrix_rank(z) != z.shape[1]:
        return None
    x, y = (v - z @ np.linalg.lstsq(z, v, rcond=None)[0] for v in (x, y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denom < 1e-12 else float(np.clip(x @ y / denom, -1, 1))


def align_people(meta, old_plan, e1_rows, voices):
    queries, by_path = historical_queries(meta, old_plan)
    require(len({r["speaker"] for r in e1_rows}) == len(e1_rows) and len(e1_rows) >= 3, "invalid E1 people")
    people, utterances, omissions = [], [], []
    for original in e1_rows:
        speaker = original["speaker"]
        require(speaker in queries and int(original["n_contexts"]) == 18, "E1 person absent from historical query contexts")
        draws = queries[speaker]
        common = draws[0] == draws[1] == draws[2]
        # Preserve actual query paths per draw rather than inventing complete cells.
        draw_metrics, unique_aligned = [], set()
        for draw in range(3):
            rows = []
            for path in sorted(draws[draw]):
                name = Path(path).stem
                if name in KNOWN_STIMULUS_ISSUES or name not in voices:
                    omissions.append({"speaker": speaker, "draw": draw, "path": path,
                                      "reason": "known_mp3_stimulus_issue" if name in KNOWN_STIMULUS_ISSUES else "missing_voice_vote"})
                    continue
                metadata, voice = by_path[path], voices[name]
                label = LABELS[metadata["label"]]
                winners = voice["winners"]
                row = {"speaker": speaker, "draw": draw, "path": path, "sentence": metadata["sentence"],
                       "acted_label": label, "voice_vote_row_id": voice["row_id"], "voice_plurality": ":".join(winners),
                       "tie": len(winners) > 1, "num_responses": voice["num_responses"],
                       "fractional_plurality_match": (1 / len(winners)) if label in winners else 0,
                       "intended_label_vote_share": voice["counts"][label] / voice["num_responses"],
                       "voice_agreement": voice["agreement"]}
                rows.append(row)
                utterances.append(row)
                unique_aligned.add(path)
            by_class = {e: [r for r in rows if r["acted_label"] == e] for e in EMOTIONS}
            require(all(by_class.values()), "aligned human covariate is missing an entire acted class")
            macro = lambda column: float(np.mean([np.mean([r[column] for r in by_class[e]]) for e in EMOTIONS]))
            draw_metrics.append({"plurality_uar_pp": 100 * macro("fractional_plurality_match"),
                                 "intended_vote_share_pp": 100 * macro("intended_label_vote_share"),
                                 "agreement_pp": 100 * macro("voice_agreement"),
                                 "n_aligned": len(rows), "class_counts": {e: len(by_class[e]) for e in EMOTIONS}})
        person = {k: (v if k == "speaker" else float(v)) for k, v in original.items()}
        require(all(np.isfinite(v) for k, v in person.items() if k != "speaker"), "invalid historical E1 values")
        person.update({"human_plurality_uar_pp": float(np.mean([m["plurality_uar_pp"] for m in draw_metrics])),
                       "human_intended_vote_share_pp": float(np.mean([m["intended_vote_share_pp"] for m in draw_metrics])),
                       "human_agreement_pp": float(np.mean([m["agreement_pp"] for m in draw_metrics])),
                       "unique_query_paths": len(set.union(*draws.values())), "unique_aligned_paths": len(unique_aligned),
                       "query_sets_identical_across_draws": common,
                       "unique_query_prompts": len({by_path[p]["sentence"] for p in set.union(*draws.values())}),
                       "draw_class_counts": json.dumps([m["class_counts"] for m in draw_metrics], sort_keys=True)})
        people.append(person)
    values = lambda k: [r[k] for r in people]
    x, y, g, h, a = map(values, ("nn3_distance", "mean_uar_pp", "global_loso_mean_distance", "human_plurality_uar_pp", "human_agreement_pp"))
    report = {"status": "completed_descriptive_historical_voice_covariate", "n_people": len(people),
              "legacy_primary_people": len(queries), "legacy_people_outside_E1": sorted(set(queries) - {r["speaker"] for r in people}),
              "n_unique_queries": len({p for r in e1_rows for paths in queries[r["speaker"]].values() for p in paths}),
              "n_unique_aligned_paths": len({r["path"] for r in utterances}), "n_aligned_path_draw_rows": len(utterances),
              "omitted_query_draw_rows": omissions, "all_query_sets_identical_across_draws": all(r["query_sets_identical_across_draws"] for r in people),
              "tie_handling": "fractional plurality match = 1/k when acted class is among k tied modes, otherwise 0; not strict unique-winner accuracy",
              "aggregation": "macro average over six acted classes after pooling actual six-rotation queries within draw, then equal average over three draws",
              "unequal_support": "known stimulus issues omitted from human covariates only; exact support by person/draw/class disclosed; E1 SER scores unchanged",
              "human_reference": "fallible rater judgments versus intended acted filename labels, not error-free true emotion",
              "descriptive_spearman_nn3_vs_historical_uar": correlation(x, y),
              "descriptive_spearman_human_plurality_vs_historical_uar": correlation(h, y),
              "descriptive_rank_partial_nn3_uar_controlling_human_plurality": correlation(x, y, [h]),
              "descriptive_rank_partial_nn3_uar_controlling_human_agreement": correlation(x, y, [a]),
              "descriptive_rank_partial_nn3_uar_controlling_human_plurality_and_global_loso": correlation(x, y, [h, g]),
              "inferential_p_values": None, "causal_interpretation": False, "main_E1_replaced": False,
              "new_SER_results_read": False, "policy_or_budget_changes": [], "formal_gate": False}
    return report, people, utterances


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus-source-root", type=Path, required=True)
    parser.add_argument("--e1-people", type=Path)
    parser.add_argument("--legacy-plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.repo.resolve() == Path(__file__).resolve().parents[2], 'executed checkout differs from --repo')
    require(bool(args.e1_people) == bool(args.legacy_plan), "provide both --e1-people and --legacy-plan or neither for source audit")
    voices, source = audit_votes(args.corpus_source_root)
    report = {"schema": "ser-voice-only-e1-description-1", "source_audit": source,
              "source_sha256": file_sha(Path(__file__)), "descriptive_only": True, "formal_gate": False,
              "E1_control": {"status": "source_audit_only"}}
    args.out.mkdir(parents=True, exist_ok=True)
    tables = {}
    if args.e1_people:
        old_plan = read_json(args.legacy_plan)
        diagnostics = read_json(args.e1_people.parent / "diagnostics.json")
        require(diagnostics["table_sha256"].get(args.e1_people.name) == file_sha(args.e1_people), "E1 table/receipt SHA mismatch")
        require(diagnostics["E1"]["input"]["legacy_plan_sha256"] == old_plan["plan_sha256"] and
                diagnostics["E1"]["input"]["legacy_plan_file_sha256"] == file_sha(args.legacy_plan), "E1/legacy plan receipt mismatch")
        meta = read_metadata(args.repo.resolve())
        report["E1_control"], people, utterances = align_people(meta, old_plan, read_csv(args.e1_people), voices)
        report["input"] = {"e1_people_sha256": file_sha(args.e1_people), "diagnostics_sha256": file_sha(args.e1_people.parent / "diagnostics.json"),
                           "legacy_plan_sha256": old_plan["plan_sha256"], "legacy_plan_file_sha256": file_sha(args.legacy_plan),
                           "manifest_sha256": meta["input"]["manifest_sha256"], "parent_plan_sha256": diagnostics["plan_sha256"]}
        for name, rows in (("voice_people.csv", people), ("voice_query_alignment.csv", utterances)):
            with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            tables[name] = file_sha(args.out / name)
    report["table_sha256"] = tables
    write_json(args.out / "voice_diagnostic.json", report)
    print(json.dumps({"source_status": source["status"], "E1_status": report["E1_control"]["status"], "formal_gate": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
