#!/usr/bin/env python3
"""Generate preregistered P0 descriptive summaries from the frozen audit table.

This script uses only the Python standard library.  It never imports or executes
candidate-repository code.  Repository-level risk means that at least one target
dataset has the risk; if none is positive but at least one is unknown, the
repository is unknown.  Low-strength rows are conservatively treated as unknown
in the main summaries, while their raw codes remain in survey_table.csv.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "survey_table.csv"
CANDIDATES = ROOT / "candidate_log.csv"
OUT = ROOT / "results" / "p0"

TARGET_DATASETS = ("RAVDESS", "CREMA-D", "IEMOCAP", "EmoDB")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def datasets_for(row: dict[str, str]) -> list[str]:
    values = [value.strip() for value in row["target_datasets"].split("|")]
    return [value for value in values if value in TARGET_DATASETS]


def relation_stratum(relation: str) -> str:
    value = relation.lower()
    if "third_party" in value or "reproduction" in value:
        return "third_party_reproduction"
    if any(token in value for token in ("official", "paper", "author", "thesis")):
        return "paper_or_author_linked"
    return "supplemental_code"


def year_stratum(value: str) -> str:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return "unknown_year"
    if year <= 2021:
        return "through_2021"
    if year <= 2023:
        return "2022_2023"
    return "2024_or_later"


def mapping(raw: str) -> dict[str, str]:
    """Parse the human-readable ``dataset=value | dataset=value`` fields."""
    result: dict[str, str] = {}
    for segment in re.split(r"\s+\|\s+", raw.strip()):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        normalized = key.strip().lower().replace("_", " ")
        result[normalized] = value.strip()
    return result


def dataset_value(row: dict[str, str], dataset: str, by_field: str, main_field: str) -> str:
    values = mapping(row.get(by_field, ""))
    keys = (dataset.lower(), dataset.lower().replace("-", ""))
    for key in keys:
        if key in values:
            return values[key]
    for key in ("all targets", "all target datasets", "all", "targets"):
        if key in values:
            return values[key]
    main = row.get(main_field, "").strip()
    return main if main != "mixed" else "unknown"


def begins(value: str, tokens: tuple[str, ...]) -> bool:
    lowered = value.strip().lower().replace("-", "_")
    return any(
        lowered == token
        or lowered.startswith(token + " ")
        or lowered.startswith(token + "_")
        or lowered.startswith(token + "(")
        for token in tokens
    )


def raw_status(domain: str, value: str) -> str:
    """Collapse a fine-grained dataset code to yes/no/unknown."""
    lowered = value.strip().lower().replace("-", "_")
    if not lowered:
        return "unknown"

    if domain == "split":
        if begins(lowered, ("unknown",)):
            return "unknown"
        if (
            begins(lowered, ("random", "utterance_random", "utterance_level_random"))
            or re.match(r"^sd\b.*\brandom\b", lowered)
            or re.match(r"^main reported sd\b.*\brandom\b", lowered)
            or "observed_overlap" in lowered
        ):
            return "yes"
        if begins(
            lowered,
            ("speaker_independent", "predefined_speaker_independent", "loso", "leave_one"),
        ):
            return "no"
        return "unknown"

    if domain == "normalization":
        if begins(lowered, ("unknown",)):
            return "unknown"
        if re.match(r"^yes(?:\b|_)", lowered):
            return "yes"
        if begins(lowered, ("no", "not_applicable")):
            return "no"
        return "unknown"

    if domain == "test_explicit":
        if begins(lowered, ("unknown",)):
            return "unknown"
        if begins(lowered, ("yes_explicit",)):
            return "yes"
        if begins(lowered, ("test_exposed_each_epoch", "no_separate_validation", "no_fixed_training")):
            return "no"
        return "unknown"

    if domain == "test_any_exposure":
        if begins(lowered, ("unknown",)):
            return "unknown"
        if begins(lowered, ("yes_explicit", "test_exposed_each_epoch")):
            return "yes"
        if begins(lowered, ("no_separate_validation", "no_fixed_training")):
            return "no"
        return "unknown"

    if domain == "augmentation":
        if begins(lowered, ("unknown",)):
            return "unknown"
        if re.match(r"^yes(?:\b|_)", lowered):
            return "yes"
        if begins(lowered, ("no_train_only", "not_applicable", "no")):
            return "no"
        return "unknown"

    if domain == "single_split_single_seed":
        if begins(lowered, ("unknown",)):
            return "unknown"
        if "single_split_single_seed" in lowered:
            return "yes"
        if begins(lowered, ("holdout", "single")):
            return "yes"
        if begins(lowered, ("one",)):
            # A single run of a multi-fold protocol is not a single split.
            if re.search(r"\b(?:5|6|10)[_ -]?fold\b|\bloso\b|leave_one", lowered):
                return "no"
            return "yes"
        if begins(
            lowered,
            ("kfold", "loso", "repeated_holdout", "5_fold", "6_fold", "10_fold", "leave_one"),
        ):
            return "no"
        return "unknown"

    if domain == "variance_absent":
        if begins(lowered, ("unknown",)):
            return "unknown"
        if begins(lowered, ("none",)):
            return "yes"
        if begins(lowered, ("sd", "se", "ci", "other")):
            return "no"
        return "unknown"

    raise ValueError(f"unknown domain: {domain}")


DOMAIN_FIELDS = {
    "split": ("split_by_dataset", "split_category"),
    "normalization": ("normalization_by_dataset", "normalization_leakage"),
    "test_explicit": ("test_selection_by_dataset", "test_selection"),
    "test_any_exposure": ("test_selection_by_dataset", "test_selection"),
    "augmentation": ("augmentation_by_dataset", "augmentation_leakage"),
    "single_split_single_seed": ("repetition_by_dataset", "evaluation_repetition"),
    "variance_absent": (None, "variance_reported"),
}


def dataset_status(row: dict[str, str], dataset: str, domain: str) -> str:
    by_field, main_field = DOMAIN_FIELDS[domain]
    value = (
        dataset_value(row, dataset, by_field, main_field)
        if by_field is not None
        else row.get(main_field, "")
    )
    status = raw_status(domain, value)
    # Some early pilot rows use the by-dataset field only for prose evidence
    # (for example ``all targets=stochastic noise ...``) while the adjacent main
    # field carries the frozen enum.  Fall back only when that main field is not
    # mixed; a mixed row without an explicit dataset code remains unknown.
    if status == "unknown" and row.get(main_field, "").strip() != "mixed":
        main_status = raw_status(domain, row.get(main_field, ""))
        if main_status != "unknown":
            status = main_status
    if row.get("evidence_strength", "").strip() == "low":
        return "unknown"
    return status


def combine(statuses: list[str]) -> str:
    if "yes" in statuses:
        return "yes"
    if "unknown" in statuses or not statuses:
        return "unknown"
    return "no"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def pct(value: float) -> str:
    return "" if math.isnan(value) else f"{100 * value:.1f}"


def interval_text(bounds: tuple[float, float]) -> str:
    if any(math.isnan(value) for value in bounds):
        return ""
    return f"{100 * bounds[0]:.1f}–{100 * bounds[1]:.1f}"


def count_summary(label: str, statuses: list[str]) -> dict[str, object]:
    counts = Counter(statuses)
    total = len(statuses)
    yes = counts["yes"]
    no = counts["no"]
    unknown = counts["unknown"]
    known = yes + no
    lower = yes / total if total else math.nan
    upper = (yes + unknown) / total if total else math.nan
    known_rate = yes / known if known else math.nan
    return {
        "metric": label,
        "N": total,
        "yes": yes,
        "no": no,
        "unknown": unknown,
        "lower_percent_unknown_as_no": pct(lower),
        "upper_percent_unknown_as_yes": pct(upper),
        "lower_wilson95_percent": interval_text(wilson(yes, total)),
        "upper_wilson95_percent": interval_text(wilson(yes + unknown, total)),
        "known_only_percent": pct(known_rate),
        "known_only_wilson95_percent": interval_text(wilson(yes, known)),
    }


def main() -> None:
    survey = read_csv(SURVEY)
    if not survey:
        raise SystemExit("survey_table.csv is empty")
    urls = [row["repository_url"] for row in survey]
    if len(urls) != len(set(urls)):
        raise SystemExit("survey_table.csv contains duplicate repository URLs")

    dataset_rows: list[dict[str, object]] = []
    repository_rows: list[dict[str, object]] = []
    for row in survey:
        datasets = datasets_for(row)
        per_domain: dict[str, list[str]] = defaultdict(list)
        for dataset in datasets:
            item: dict[str, object] = {
                "repository_id": row["repository_id"],
                "repository_url": row["repository_url"],
                "dataset": dataset,
                "relation_stratum": relation_stratum(row["repo_relation"]),
                "publication_year": row["publication_year"] or "unknown",
                "year_stratum": year_stratum(row["publication_year"]),
                "evidence_strength": row["evidence_strength"],
            }
            for domain in DOMAIN_FIELDS:
                by_field, main_field = DOMAIN_FIELDS[domain]
                item[f"{domain}_source_value"] = (
                    dataset_value(row, dataset, by_field, main_field)
                    if by_field is not None
                    else row.get(main_field, "")
                )
                status = dataset_status(row, dataset, domain)
                item[domain] = status
                per_domain[domain].append(status)
            dataset_rows.append(item)

        repository_item: dict[str, object] = {
            "repository_id": row["repository_id"],
            "repository_url": row["repository_url"],
            "relation_stratum": relation_stratum(row["repo_relation"]),
            "publication_year": row["publication_year"] or "unknown",
            "year_stratum": year_stratum(row["publication_year"]),
            "evidence_strength": row["evidence_strength"],
            "target_datasets": row["target_datasets"],
        }
        for domain in DOMAIN_FIELDS:
            repository_item[domain] = combine(per_domain[domain])
        repository_rows.append(repository_item)

    risk_fields = list(DOMAIN_FIELDS)
    summary_rows = [
        count_summary(field, [str(row[field]) for row in repository_rows])
        for field in risk_fields
    ]

    strata_rows: list[dict[str, object]] = []
    strata_specs = {
        "relation": sorted({str(row["relation_stratum"]) for row in repository_rows}),
        "year_bin": sorted({str(row["year_stratum"]) for row in repository_rows}),
        "publication_year": sorted({str(row["publication_year"]) for row in repository_rows}),
    }
    for stratum_type, levels in strata_specs.items():
        key = {
            "relation": "relation_stratum",
            "year_bin": "year_stratum",
            "publication_year": "publication_year",
        }[stratum_type]
        for level in levels:
            subset = [row for row in repository_rows if row[key] == level]
            for field in risk_fields:
                result = count_summary(field, [str(row[field]) for row in subset])
                result = {"stratum_type": stratum_type, "stratum": level, **result}
                strata_rows.append(result)

    dataset_summary_rows: list[dict[str, object]] = []
    for dataset in TARGET_DATASETS:
        subset = [row for row in dataset_rows if row["dataset"] == dataset]
        for field in DOMAIN_FIELDS:
            result = count_summary(field, [str(row[field]) for row in subset])
            result = {"dataset": dataset, **result}
            dataset_summary_rows.append(result)

    candidate_rows = read_csv(CANDIDATES)
    status_counts = Counter(row["screening_status"] for row in candidate_rows)
    exclusion_counts = Counter(
        row["exclusion_code"] for row in candidate_rows
        if row["screening_status"] == "exclude"
    )
    screened_values: list[int] = []
    screened_na = 0
    query_new_values: list[int] = []
    for search in read_csv(ROOT / "search_log.csv"):
        try:
            screened_values.append(int(search["results_screened"]))
        except ValueError:
            screened_na += 1
        try:
            query_new_values.append(int(search["new_candidates"]))
        except ValueError:
            pass
    decided = status_counts["include"] + status_counts["exclude"]
    flow_rows: list[dict[str, object]] = [
        {"flow_type": "search_and_screening_flow", "category": "search_requests_total", "count": len(read_csv(ROOT / "search_log.csv"))},
        {"flow_type": "search_and_screening_flow", "category": "search_requests_non_numeric_result_count", "count": screened_na},
        {"flow_type": "search_and_screening_flow", "category": "result_slots_screened_sum_duplicates_and_replays_included", "count": sum(screened_values)},
        {"flow_type": "search_and_screening_flow", "category": "query_reported_new_candidates_sum_before_global_dedup", "count": sum(query_new_values)},
        {"flow_type": "search_and_screening_flow", "category": "deduplicated_candidate_urls", "count": len(candidate_rows)},
        {"flow_type": "search_and_screening_flow", "category": "candidates_with_final_screening_decision", "count": decided},
        {"flow_type": "search_and_screening_flow", "category": "included_audited", "count": status_counts["include"]},
        {"flow_type": "search_and_screening_flow", "category": "excluded_with_direct_evidence", "count": status_counts["exclude"]},
        {"flow_type": "search_and_screening_flow", "category": "pending_not_fully_screened", "count": status_counts["pending"]},
    ]
    flow_rows.extend(
        {"flow_type": "screening_status", "category": key, "count": value}
        for key, value in sorted(status_counts.items())
    )
    flow_rows.extend(
        {"flow_type": "exclusion_code", "category": key, "count": value}
        for key, value in sorted(exclusion_counts.items())
    )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT / "repository_risk.csv",
        repository_rows,
        [
            "repository_id", "repository_url", "relation_stratum", "year_stratum",
            "publication_year", "evidence_strength", "target_datasets", *risk_fields,
        ],
    )
    write_csv(
        OUT / "repository_dataset_risk.csv",
        dataset_rows,
        [
            "repository_id", "repository_url", "dataset", "relation_stratum",
            "publication_year", "year_stratum", "evidence_strength",
            *[
                field
                for domain in DOMAIN_FIELDS
                for field in (f"{domain}_source_value", domain)
            ],
        ],
    )
    summary_fields = [
        "metric", "N", "yes", "no", "unknown",
        "lower_percent_unknown_as_no", "upper_percent_unknown_as_yes",
        "lower_wilson95_percent", "upper_wilson95_percent",
        "known_only_percent", "known_only_wilson95_percent",
    ]
    write_csv(OUT / "summary_counts.csv", summary_rows, summary_fields)
    write_csv(
        OUT / "strata_counts.csv",
        strata_rows,
        ["stratum_type", "stratum", *summary_fields],
    )
    write_csv(
        OUT / "dataset_counts.csv",
        dataset_summary_rows,
        ["dataset", *summary_fields],
    )
    write_csv(OUT / "candidate_flow.csv", flow_rows, ["flow_type", "category", "count"])

    print(
        f"repositories={len(repository_rows)} repository_datasets={len(dataset_rows)} "
        f"candidates={len(candidate_rows)} output={OUT}"
    )


if __name__ == "__main__":
    main()
