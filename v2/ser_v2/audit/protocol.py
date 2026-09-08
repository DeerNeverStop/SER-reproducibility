"""Frozen audit protocol constants: eligibility codes, lineage rule, endpoint dictionary,
rater form schema, and caps."""
from __future__ import annotations

ELIGIBILITY_CODES = {
    "E1": "not an SER task",
    "E2": "no evaluation on RAVDESS, CREMA-D, IEMOCAP or EmoDB",
    "E3": "no relevant training/evaluation code (paper-only, data-only, empty)",
    "E4": "repository unavailable at the pinned snapshot",
    "E5": "duplicate or materially unchanged fork lineage (lineage rule)",
    "E6": "unscreenable within the 10-minute screening cap (recorded, ineligible)",
}

LINEAGE_RULE = {
    "version": "v2-1",
    "merge_if_any": [
        "explicit fork/parent relation in repository metadata",
        "identical root tree SHA at the pinned default-branch commit",
        "identical content-blob SHA set",
        "executable-blob containment >= 0.90 (closed suffix list; notebook = one blob; dedup by blob SHA) AND a corroborator (same project name, identical root README blob, or owner lineage)",
    ],
    "keeper": "unique upstream, else earliest created_at, else canonical URL order",
    "executable_suffixes": [".py", ".ipynb", ".sh", ".m", ".r", ".jl", ".lua", ".js", ".ts", ".cfg", ".yaml", ".yml", ".json", ".toml"],
    "fixed_before_screening": True,
}

ENDPOINTS = {
    "Y_test": "an active train -> held-out path whose metric is presented as the final reported result permits same-speaker membership on both sides (dual use recorded in descriptor column)",
    "Y_val": "only a train -> validation/model-selection path permits same-speaker membership AND a separately reported test set is verified speaker-exclusive",
    "N": "every active path verified speaker-exclusive and path enumeration closed",
    "U": "undecidable, review cap exceeded, or no applicable train/evaluation split",
}
PRIMARY = "Y_any = Y_test or Y_val"
SECONDARY = "Y_test"

DESCRIPTOR_COLUMNS = [
    "test_set_used_for_selection (Y/N/U)", "single_split_no_variance (Y/N/U)",
    "dialogue_partner_contamination_possible (IEMOCAP session split) (Y/N/U/NA)", "corpus_set", "combined_corpus_loader (Y/N)",
    "class_subset", "notebook_vs_script", "framework", "year", "reported_top_line_metric", "execution_eligible (Y/N)",
    "prior_lineage_membership_and_historical_verdict (filled after adjudication; hidden from raters)",
]

CAPS = {"screening_minutes_per_candidate": 10, "review_hours_per_unit_per_rater": 2.0, "sample_n": 60,
        "double_coded_units": 30, "human_first_units_from_unseen_lineages": True}

RATER_FORM_SCHEMA = {
    "type": "object",
    "required": ["unit_id", "rater", "commit_sha", "verdict", "evidence", "review_minutes", "enumeration_closed"],
    "properties": {
        "unit_id": {"type": "string"}, "rater": {"type": "string", "enum": ["H", "AI"]}, "commit_sha": {"type": "string"},
        "verdict": {"type": "string", "enum": ["Y_test", "Y_val", "N", "U"]},
        "evidence": {"type": "array", "items": {"type": "object", "required": ["path", "line_start", "line_end", "fragment_sha256", "role"],
                                                 "properties": {"path": {"type": "string"}, "line_start": {"type": "integer"}, "line_end": {"type": "integer"},
                                                                "fragment_sha256": {"type": "string"}, "role": {"type": "string", "enum": ["entry_point", "split_statement", "evaluation", "selection"]}}}},
        "review_minutes": {"type": "number"}, "enumeration_closed": {"type": "boolean"},
        "u_reason": {"type": "string"}, "descriptors": {"type": "object"},
    },
}
