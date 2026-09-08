"""Format accepted final-program outputs into one shared bilingual fact source.

No prediction loading, averaging, inference, checkpoint selection or plotting.
Admission shares plot_results.load, then cross-checks the complete 384-unit
commitments against the independent audit. Historical full-weight acceptance is
not rerun here. --synthetic accepts only explicitly marked synthetic fixtures.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from . import plot_results


SCHEMA = "ser-final-program-paper-facts-1"
PROGRAM = "SER26-FINAL-CHECKPOINT-PROGRAM-1"
CORPORA = ("cremad", "subesco", "ravdess")
NAMES = {"cremad": "CREMA-D", "subesco": "SUBESCO", "ravdess": "RAVDESS"}
CLASSES = {"cremad": 6, "subesco": 7, "ravdess": 8}
RULES = ("seen_ce", "unseen_ce", "seen_uar", "unseen_uar", "last")
HASH = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_accepted(results, audit_path, *, synthetic=False):
    """Use the existing figure gate and bind its full unit inventory; no logits."""
    results, audit_path = Path(results).resolve(), Path(audit_path).resolve()
    result, _, pins = plot_results.load(results, audit_path)
    gate = read_json(results / "complete_gate.json")
    audit = read_json(audit_path)
    require(result["schema"] == "ser-final-program-score-1" and result["program"] == PROGRAM,
            "wrong scientific result schema/program")
    require(gate["schema"] == "ser-final-program-complete-gate-1"
            and audit["schema"] == "ser-final-program-independent-numeric-audit-1"
            and audit["phase"] == "formal", "wrong complete-gate/audit contract")
    require(gate["plan_sha256"] == result["plan_sha256"] == audit["plan_sha256"]
            and gate["lock_sha256"] == result["inputs"]["lock_sha256"] == audit["lock_sha256"],
            "result/gate/audit plan or lock differs")
    require(audit["main_result_sha256"] == plot_results.sha(results / "results.json")
            and audit["main_result_semantic_sha256"] == result["result_sha256"],
            "audit result binding differs")
    require(audit["numeric_values_checked"] > 0 and audit["new_fits"] == 0
            and audit["model_inference_performed"] is False, "not a completed numerical-only audit")
    require(audit["frozen_sources"]["v3/final_program_20260907/score.py"]
            == result["inputs"]["score_source_sha256"], "frozen scorer identity differs")
    require(audit["tests_checked"] == [dict(corpus=t["corpus"], estimand=t["estimand"])
                                      for t in result["tests"]], "audited family differs")
    actual_phase = Path(audit["relocation"]["actual_run_dir"]).resolve()
    require(audit["relocation"]["recorded_run_dir"] == result["inputs"]["run_dir"],
            "audit recorded phase differs")
    input_hashes = result["inputs"]["byte_sha256"]
    audit_pins = audit["source_and_input_sha256"]
    require(input_hashes["COMPLETE_GATE.json"] == audit["score_gate_sha256"]
            and input_hashes["ledger.jsonl"] == gate["ledger_sha256"], "score top-level closure differs")
    expected_input_keys = {"SOURCE_LOCK.json", "plan_snapshot.json", "COMPLETE_GATE.json", "ledger.jsonl"}
    for name in expected_input_keys - {"COMPLETE_GATE.json"}:
        require(audit_pins.get(str(actual_phase / name)) == input_hashes[name], "top-level audit pin differs")
    ids = set(gate["done_sha256"])
    require(len(ids) == 384 and ids == set(gate["artifacts_sha256"])
            == set(audit["checkpoint_gate_pins_not_rehashed"]), "384-unit identity inventory incomplete")
    for uid in ids:
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", uid) and uid not in (".", ".."), "unsafe committed unit ID")
        done_key = f"units/{uid}/DONE"
        expected_input_keys.add(done_key)
        done_hash = gate["done_sha256"][uid]
        require(HASH.fullmatch(done_hash) and input_hashes.get(done_key) == done_hash
                and audit_pins.get(str(actual_phase / done_key)) == done_hash, "unit DONE audit binding differs")
        artifacts = gate["artifacts_sha256"][uid]
        require(len(artifacts) == 4, "unit does not bind all four artifacts")
        attempts, basenames = set(), set()
        for relative, expected_hash in artifacts.items():
            require(re.fullmatch(r"attempts/\d{4}/(checkpoint\.pt|predictions\.npz|history\.json|receipt\.json)", relative)
                    and HASH.fullmatch(expected_hash), "unsafe/incomplete artifact identity")
            attempt, name = relative.rsplit("/", 1)
            attempts.add(attempt); basenames.add(name)
            key = f"units/{uid}/{relative}"
            if name == "checkpoint.pt":
                cp = audit["checkpoint_gate_pins_not_rehashed"][uid]
                require(cp["gate_sha256"] == expected_hash and cp["rehash_performed"] is False,
                        "historical checkpoint binding differs")
            else:
                require(audit_pins.get(str(actual_phase / key)) == expected_hash,
                        "small-artifact audit binding differs")
                if name in ("predictions.npz", "receipt.json"):
                    expected_input_keys.add(key)
                    require(input_hashes.get(key) == expected_hash, "score artifact binding differs")
        require(len(attempts) == 1 and basenames == {"checkpoint.pt", "predictions.npz", "history.json", "receipt.json"},
                "artifact attempt closure differs")
    require(set(input_hashes) == expected_input_keys, "score input inventory has missing/extra members")
    for name in plot_results.COUNTS:
        path = results / (name + ".csv")
        require(audit_pins.get(str(path)) == plot_results.sha(path), "CSV is not this audit's input")
    marker = "SYNTHETIC" in json.dumps({"analysis": result["analysis_contract"],
                                        "scope": gate["scope"], "commit": audit["source_commit"]}).upper()
    require(marker is bool(synthetic), "synthetic fixtures require --synthetic; formal mode must have no fixture marker")
    pins[str(Path(__file__).resolve())] = plot_results.sha(__file__)
    return result, audit, gate, pins, actual_phase


def make_facts(result, audit, gate, pins, *, synthetic=False):
    """Copy complete recorded summaries; no derived scientific numbers."""
    require(set(result["corpora"]) == set(CORPORA), "native corpora missing")
    for corpus in CORPORA:
        item = result["corpora"][corpus]
        require(item["main_trajectories"] == 120 and item["n_draws"] == 24 and item["folds_per_draw"] == 5,
                "corpus aggregation support differs")
        require(set(item["rules"]) == set(RULES), "must retain four rules and last")
        for rule in RULES:
            counts = item["rules"][rule]["selected_epoch_counts"]
            require(sum(counts.values()) == 120 and all(str(int(e)) == e and 1 <= int(e) <= 15 for e in counts),
                    "selected epoch distribution is incomplete")
    for test in result["tests"]:
        require(test["n_draws"] == 24 and test["df"] == 23, "inference unit differs")
        if test["status"] == "undefined_t_zero_sample_variance":
            require(test["p_two_sided"] is None and test["holm_p_six"] is None
                    and test["pointwise_95_ci_pp"] is None, "undefined test must remain NA")
        else:
            require(test["status"] == "estimated" and len(test["pointwise_95_ci_pp"]) == 2
                    and test["p_two_sided"] is not None and test["holm_p_six"] is not None,
                    "unknown test state")
    require(result["control"]["n_pairs"] == result["control"]["n_added_B_fits"] == 24
            and result["control"]["inference"] is False, "role control is not the prescribed description")
    return dict(schema=SCHEMA,
        data_status="SYNTHETIC_DEMONSTRATION_NOT_RESEARCH_EVIDENCE" if synthetic else "ACCEPTED_FORMAL_RESULT_SUMMARIES",
        program=result["program"], plan_sha256=result["plan_sha256"], lock_sha256=gate["lock_sha256"],
        source_result_semantic_sha256=result["result_sha256"], source_audit_semantic_sha256=audit["audit_sha256"],
        source_and_input_sha256=dict(pins), counts=deepcopy(result["counts"]),
        native_class_counts=CLASSES, analysis_contract=deepcopy(result["analysis_contract"]),
        units=dict(absolute_outer_uar="percent (0--100), whole-role native-class macro recall",
                   effects="percentage points (pp), including delta_CE: this is an outer-UAR difference, not a CE loss difference",
                   epoch="one-based training epoch", epoch_counts="counts among 120 main A trajectories per corpus",
                   agreement="fraction (0--1)", p_values="dimensionless; JSON null renders as NA"),
        inference=dict(family="six prespecified tests: delta_CE and J within each of three native corpora",
                       confidence_intervals="recorded pointwise 95% t intervals, df=23; not simultaneous or Holm-adjusted intervals",
                       p_values="recorded two-sided t p and six-family Holm p; undefined remains NA, not 1 or 0",
                       sampling="five-fold mean per draw, then 24 equally weighted draws; conditional on this finite corpus and program",
                       descriptive="delta_UAR, all absolute UAR, epochs, agreement, middle/late, oracle and E: no new CI or p"),
        source_values=deepcopy({key: result[key] for key in ("tests", "corpora", "control")}),
        interpretation=deepcopy(result["interpretation"]),
        formatting=dict(decimals=6, p_significant_digits=6, source_floats_preserved_in_json=True),
        scope=dict(new_training=0, new_model_inference=0, new_statistical_tests=0,
                   new_scientific_statistics=0, selection_or_aggregation_recomputed=False,
                   acceptance=("SYNTHETIC fixture exercises the 384-unit admission interface with a declared gate stub; not real execution acceptance."
                               if synthetic else "Shared plot-results gate plus full 384-unit audit commitments; no current weight rehash or numerical replay is performed here.")))


def number(value, *, p=False):
    if value is None:
        return "NA"
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), "nonfinite fact")
    return format(value, ".6g" if p else ".6f")


def markdown(facts, language):
    zh = language == "zh"
    require(language in ("zh", "en"), "unknown language")
    values = facts["source_values"]
    lines = ["# 正式程序共用事实表" if zh else "# Shared facts for the completed program", ""]
    if facts["data_status"].startswith("SYNTHETIC"):
        lines += ["**SYNTHETIC — 仅演练，不能引用为真实研究结果。**" if zh else
                  "**SYNTHETIC — Demonstration only; not research evidence.**", ""]
    if facts["data_status"].startswith("SYNTHETIC"):
        intro = ("本页仅用已标记的模拟结果与模拟 gate 演练完整接口，不代表真实训练验收。JSON 保存原始精度；表中显示六位小数。" if zh else
                 "This page exercises the complete interface using explicitly synthetic results and a gate stub; it is not real training acceptance. JSON preserves source precision; tables display six decimals.")
    else:
        intro = ("所有数字直接复制自同一份已绑定完整 384 单元验收与独立数值审计的结果；保留全部预定对照。JSON 保存原始精度，表中显示六位小数。" if zh else
                 "All numbers are copied from one result bound to complete 384-unit acceptance and an independent numerical audit. All prespecified comparisons are retained. JSON preserves source precision; tables display six decimals.")
    lines += [f"Plan SHA-256: `{facts['plan_sha256']}`", "", intro, "",
        "## 六项预定检验" if zh else "## Six prespecified tests", "",
        "δCE = T(seen-CE) − T(unseen-CE)；δUAR = T(seen-UAR) − T(unseen-UAR)；J = δCE − δUAR。T 是外测 UAR，单位为百分点（pp）；δCE 不是交叉熵损失差，也不是 V−T 乐观偏差。" if zh else
        "δCE = T(seen-CE) − T(unseen-CE); δUAR = T(seen-UAR) − T(unseen-UAR); J = δCE − δUAR. T is outer-test UAR and differences are percentage points (pp). δCE is neither a CE-loss difference nor V−T optimism.", "",
        "| 语料 | 效应 | 均值（pp） | 逐项 95% t CI（pp） | 双侧原始 p | 六项 Holm p | 状态 |" if zh else
        "| Corpus | Effect | Mean (pp) | Pointwise 95% t CI (pp) | Raw two-sided p | Six-test Holm p | Status |",
        "|---|---|---:|---|---:|---:|---|"]
    lookup = {(t["corpus"], t["estimand"]): t for t in values["tests"]}
    for corpus in CORPORA:
        for estimand in ("delta_CE", "J"):
            t = lookup[corpus, estimand]
            ci = t["pointwise_95_ci_pp"]
            interval = "NA" if ci is None else f"[{number(ci[0])}, {number(ci[1])}]"
            lines.append(f"| {NAMES[corpus]} | {'δCE' if estimand == 'delta_CE' else 'J'} | {number(t['mean_pp'])} | {interval} | {number(t['p_two_sided'],p=True)} | {number(t['holm_p_six'],p=True)} | {t['status']} |")
    lines += ["", "每项均为 24 个完整 draw，df=23；每个 draw 先平均五折。区间是逐项 95% t 区间，不是同时区间；Holm 只调整六项检验的 p。零样本方差导致未定义的检验及区间明确记 NA，不能解释成等效。推断条件于有限语料与固定程序。" if zh else
        "Each test uses 24 complete draws, df=23, after averaging five folds per draw. Intervals are pointwise 95% t intervals, not simultaneous intervals; Holm adjusts only the six p values. Undefined zero-variance tests and intervals remain NA and do not establish equivalence. Inference is conditional on the finite corpus and fixed program.", "",
        "## 外测绝对 UAR 与 δUAR（仅描述）" if zh else "## Absolute outer UAR and δUAR (descriptive)", "",
        "| 语料（原生类数） | seen-CE UAR (%) | unseen-CE UAR (%) | seen-UAR UAR (%) | unseen-UAR UAR (%) | last-15 UAR (%) | δUAR（pp） |" if zh else
        "| Corpus (native classes) | seen-CE UAR (%) | unseen-CE UAR (%) | seen-UAR UAR (%) | unseen-UAR UAR (%) | last-15 UAR (%) | δUAR (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus in CORPORA:
        c = values["corpora"][corpus]
        row = [f"{NAMES[corpus]} ({CLASSES[corpus]})"] + [number(c["rules"][rule]["mean_outer_uar"]) for rule in RULES]
        lines.append("| " + " | ".join(row + [number(c["delta_UAR_descriptive_mean_pp"])]) + " |")
    lines += ["", "每库 120 条主训练轨迹（24 draw × 5 fold），四种规则从同一轨迹选择检查点；last 固定第 15 轮。本表 CI 与 p 均为 NA（未检验）。不同语料的类别数与任务不同，绝对分数不是语言效应或公平难度排名。" if zh else
        "Each corpus has 120 main trajectories (24 draws × 5 folds). The four rules select checkpoints from the same trajectory; last is fixed at epoch 15. CI and p are NA (not tested) for this table. Native class sets/tasks differ, so absolute scores do not identify language effects or rank corpus difficulty fairly.", "",
        "## 检查点轮次与 oracle 差距（仅描述）" if zh else "## Checkpoint epochs and oracle shortfall (descriptive)", "",
        "| 语料 | 规则 | 平均选中轮次 | 轮次:次数（合计 120） | 平均 oracle shortfall（pp） |" if zh else
        "| Corpus | Rule | Mean selected epoch | Epoch:count (total 120) | Mean oracle shortfall (pp) |",
        "|---|---|---:|---|---:|"]
    for corpus in CORPORA:
        for rule in RULES:
            s = values["corpora"][corpus]["rules"][rule]
            counts = "; ".join(f"{e}:{n}" for e, n in sorted(s["selected_epoch_counts"].items(), key=lambda x: int(x[0])))
            lines.append(f"| {NAMES[corpus]} | {rule} | {number(s['mean_selected_epoch'])} | {counts} | {number(s['mean_oracle_shortfall_pp'])} |")
    lines += ["", "oracle shortfall 是同一轨迹全部 15 轮最高外测 UAR 减去该规则检查点的外测 UAR；oracle 从未用于实际保留检查点的选择。该表无 CI/p（NA），不能据此事后选轮次或改主终点。" if zh else
        "Oracle shortfall is the trajectory's maximum outer UAR across all 15 epochs minus outer UAR at the rule's checkpoint. The oracle never selects a retained checkpoint. CI/p are NA; these descriptions do not authorize post hoc epoch or primary-endpoint changes.", "",
        "## 其他预定描述" if zh else "## Other prespecified descriptions", "",
        "| 语料 | 8–10 轮均值 − 14–15 轮均值（pp） | CE 规则选轮一致比例 | UAR 规则选轮一致比例 |" if zh else
        "| Corpus | Mean epochs 8–10 minus mean epochs 14–15 (pp) | CE-rule epoch agreement (fraction) | UAR-rule epoch agreement (fraction) |",
        "|---|---:|---:|---:|"]
    for corpus in CORPORA:
        c = values["corpora"][corpus]
        lines.append("| " + " | ".join([NAMES[corpus], number(c["middle_8_10_minus_late_14_15_descriptive_mean_pp"]),
                                           number(c["ce_epoch_agreement_fraction"]), number(c["uar_epoch_agreement_fraction"])]) + " |")
    control = values["control"]
    lines += ["", "本表使用完整轨迹的预定早晚轮次摘要；一致比例单位为 0–1，不是百分比。均无 CI/p（NA）。" if zh else
        "This table uses the prespecified middle/late summary of complete trajectories. Agreement is a fraction (0–1), not a percentage. CI/p are NA (not tested).", "",
        "| 末轮角色交换对照 | 配对数 | 额外 B 训练数 | E 均值（pp） | CI | p |" if zh else
        "| Last-epoch role-swap control | Pairs | Additional B fits | Mean E (pp) | CI | p |",
        "|---|---:|---:|---:|---|---|",
        f"| CREMA-D, fold 0, epoch 15 | {control['n_pairs']} | {control['n_added_B_fits']} | {number(control['E_mean_pp'])} | NA | NA |", "",
        "E = 0.5 × [(QA(MA15) − QA(MB15)) + (QB(MB15) − QB(MA15))]。这是 24 个预定 fold-0 成对整组训练人群替换的描述，不是单个说话人的因果效应。query 同时供主 A 轨迹的检查点选择使用，不能称其从未参与任何选择。" if zh else
        "E = 0.5 × [(QA(MA15) − QA(MB15)) + (QB(MB15) − QB(MA15))]. This describes 24 prespecified fold-0 pairs replacing whole training groups, not an individual speaker causal effect. Query data also serve checkpoint selection for main A trajectories; they are not globally unused for selection.", ""]
    return "\n".join(lines)


def execute(results, audit, out, *, synthetic=False):
    results, audit, out = Path(results).resolve(), Path(audit).resolve(), Path(out).resolve()
    require(not out.exists(), "facts output must be a new directory")
    for protected in (results, audit.parent):
        require(not out.is_relative_to(protected) and not protected.is_relative_to(out), "facts output overlaps inputs")
    result, audit_record, gate, pins, phase = load_accepted(results, audit, synthetic=synthetic)
    for protected in (phase, Path(result["inputs"]["run_dir"]).resolve()):
        require(not out.is_relative_to(protected) and not protected.is_relative_to(out), "facts must not enter/contain frozen phase")
    facts = make_facts(result, audit_record, gate, pins, synthetic=synthetic)
    rendered = {f"tables_{language}.md": markdown(facts, language).encode("utf-8") for language in ("zh", "en")}
    facts["rendered_files_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in rendered.items()}
    facts["facts_sha256"] = digest(facts)
    rendered["facts.json"] = (json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for path, expected in pins.items():
        require(plot_results.sha(path) == expected, "input/source changed during formatting")
    out.mkdir(parents=True, exist_ok=False)
    for name, data in rendered.items():
        with (out / name).open("xb") as stream:
            stream.write(data)
    for path, expected in pins.items():
        require(plot_results.sha(path) == expected, "input/source changed during output")
    return {"pass": True, "schema": SCHEMA, "data_status": facts["data_status"], "files": len(rendered),
            "facts_sha256": facts["facts_sha256"], "new_tests": 0, "new_scientific_statistics": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true", help="Only explicitly marked fixture inputs; watermark every output")
    args = parser.parse_args()
    result = execute(args.results, args.audit, args.out, synthetic=args.synthetic)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
