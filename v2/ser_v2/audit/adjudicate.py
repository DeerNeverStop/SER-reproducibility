"""Scripted adjudication between rater H and rater AI, and the estimation-only statistics."""
from __future__ import annotations

import json
from pathlib import Path

from ..stats import cohen_kappa, frame_envelope, kappa_bootstrap_ci, partial_identification, wilson

Y_LEVELS = ("Y_test", "Y_val", "N", "U")


def chain_complete(record: dict) -> bool:
    """A Y/N verdict needs an entry point, a split statement with fragment hashes and closed enumeration."""
    roles = {e.get("role") for e in record.get("evidence", [])}
    hashed = all(e.get("fragment_sha256") and e.get("path") for e in record.get("evidence", []))
    return bool(record.get("enumeration_closed")) and "entry_point" in roles and "split_statement" in roles and hashed


def adjudicate(h: dict | None, ai: dict) -> dict:
    """Rules (frozen): identical verdicts stand; a Y/N verdict without a complete evidence chain
    becomes U; if both are Y-type but differ on test vs val, take Y_test when either chain shows a
    reported held-out path, else Y_val; any other disagreement -> U for a second human to resolve."""
    if h is None:
        v = ai["verdict"] if (ai["verdict"] == "U" or chain_complete(ai)) else "U"
        return {"verdict": v, "rule": "single-rated (AI + author review)", "needs_second_human": False}
    if h["verdict"] == ai["verdict"]:
        if h["verdict"] in ("Y_test", "Y_val", "N") and not (chain_complete(h) or chain_complete(ai)):
            return {"verdict": "U", "rule": "agreement without a complete chain -> U", "needs_second_human": False}
        return {"verdict": h["verdict"], "rule": "agreement", "needs_second_human": False}
    ys = {h["verdict"], ai["verdict"]}
    if ys == {"Y_test", "Y_val"}:
        v = "Y_test" if any(chain_complete(r) and r["verdict"] == "Y_test" for r in (h, ai)) else "Y_val"
        return {"verdict": v, "rule": "Y-type disagreement resolved by chain", "needs_second_human": False}
    return {"verdict": "U", "rule": "semantic disagreement -> U pending second human", "needs_second_human": True}


def audit_statistics(verdicts: dict[str, str], frame_n: int, double_coded: dict[str, tuple[str, str]] | None = None,
                     execution: dict[str, dict] | None = None) -> dict:
    n = len(verdicts)
    y_test = sum(1 for v in verdicts.values() if v == "Y_test")
    y_val = sum(1 for v in verdicts.values() if v == "Y_val")
    u = sum(1 for v in verdicts.values() if v == "U")
    y_any = y_test + y_val
    out = {"n": n, "Y_test": y_test, "Y_val": y_val, "Y_any": y_any, "N": n - y_any - u, "U": u,
           "primary_Y_any": {"point": y_any / n, "wilson95": wilson(y_any, n), "partial_identification": partial_identification(y_any, u, n),
                             "frame_envelope": frame_envelope(frame_n, n, y_any, u)},
           "secondary_Y_test": {"point": y_test / n, "wilson95": wilson(y_test, n), "partial_identification": partial_identification(y_test, u, n),
                                "frame_envelope": frame_envelope(frame_n, n, y_test, u)},
           "Y_val_only_fraction": y_val / n}
    if double_coded:
        a = [double_coded[k][0] for k in sorted(double_coded)]
        b = [double_coded[k][1] for k in sorted(double_coded)]
        out["reliability"] = {"n_double_coded": len(a), "kappa_4level": cohen_kappa(a, b), "kappa_4level_ci": kappa_bootstrap_ci(a, b),
                              "kappa_Yany_vs_not": cohen_kappa([x.startswith("Y") for x in a], [x.startswith("Y") for x in b]),
                              "note": "reported, never a gate; human-AI agreement unless a second human coded"}
    if execution:
        elig = [k for k, e in execution.items() if e.get("eligible")]
        overlaps = [e["realized_overlap_median"] for k, e in execution.items() if e.get("eligible") and "realized_overlap_median" in e]
        discord = [k for k in elig if (verdicts.get(k, "").startswith("Y") and execution[k].get("realized_overlap_max", 0) == 0)
                   or (verdicts.get(k) == "N" and execution[k].get("realized_overlap_max", 0) > 0)]
        out["execution_tier"] = {"eligible_fraction": len(elig) / max(1, n), "n_eligible": len(elig),
                                 "realized_overlap_median_over_eligible": (sorted(overlaps)[len(overlaps) // 2] if overlaps else None),
                                 "discordant_units": discord, "scope": "describes the execution-eligible subset only; never generalized to all units"}
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Adjudicate rater records and compute audit statistics")
    ap.add_argument("--records", type=Path, required=True, help="JSON: {unit_id: {'H': record|null, 'AI': record}}")
    ap.add_argument("--frame-n", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    recs = json.loads(a.records.read_text(encoding="utf-8"))
    adj = {k: adjudicate(v.get("H"), v["AI"]) for k, v in recs.items()}
    verdicts = {k: v["verdict"] for k, v in adj.items()}
    double = {k: (v["H"]["verdict"], v["AI"]["verdict"]) for k, v in recs.items() if v.get("H")}
    stats = audit_statistics(verdicts, a.frame_n, double or None)
    a.out.write_text(json.dumps({"adjudication": adj, "statistics": stats}, indent=1), encoding="utf-8")
    print(json.dumps(stats["primary_Y_any"]))


if __name__ == "__main__":
    main()
