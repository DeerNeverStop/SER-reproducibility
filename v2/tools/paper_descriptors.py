"""Descriptor macros for the manuscript (\\newcommand, digits spelled out in names), generated from
results.json and the descriptor JSON files so that no number in the text is typed by hand.

    python -m tools.paper_descriptors --run runs/main --out paper/v2/descriptor_macros.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import read_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    R = read_json(a.run / "results.json")
    D = R["descriptors"]
    M = {}

    def put(name, value, nd=2):
        if value is None:
            M[name] = "--"
        elif isinstance(value, (int, np.integer)):
            M[name] = str(int(value))
        else:
            M[name] = f"{float(value):.{nd}f}"

    # D01 allocation package
    for lv, key in (("ravdess", "Rav"), ("cremad", "Cre"), ("subesco_980", "Sub")):
        put(f"DOneScratch{key}", D["D01"][lv]["scratch"]["mean"]); put(f"DOneProbe{key}", D["D01"][lv]["probe"]["mean"])
    # D03 corpus differences
    for k, key in (("ravdess_minus_cremad", "RavCre"), ("ravdess_minus_subesco_980", "RavSub")):
        put(f"DThree{key}", D["D03"][k]["diff"]); put(f"DThree{key}Lo", D["D03"][k]["ci_low"]); put(f"DThree{key}Hi", D["D03"][k]["ci_high"])
    # D04 panels
    put("DFourCells", D["D04"]["sub_1400_one_minus_sub_700_one"]["mean"]); put("DFourCellsLo", D["D04"]["sub_1400_one_minus_sub_700_one"]["ci_low"]); put("DFourCellsHi", D["D04"]["sub_1400_one_minus_sub_700_one"]["ci_high"])
    put("DFourTakes", D["D04"]["sub_700_two_minus_sub_700_one"]["mean"]); put("DFourTakesLo", D["D04"]["sub_700_two_minus_sub_700_one"]["ci_low"]); put("DFourTakesHi", D["D04"]["sub_700_two_minus_sub_700_one"]["ci_high"])
    # D05
    put("DFive", D["D05"]["mean"]); put("DFiveLo", D["D05"]["ci_low"]); put("DFiveHi", D["D05"]["ci_high"])
    # D06
    for lv, key in (("cremad", "Cre"), ("subesco_full", "Sub")):
        d = D["D06"][lv]
        put(f"DSixBoth{key}", d["both_minus_none"]["mean"]); put(f"DSixInter{key}", d["interaction"]["mean"]); put(f"DSixInter{key}Lo", d["interaction"]["ci_low"]); put(f"DSixInter{key}Hi", d["interaction"]["ci_high"])
        put(f"DSixGOne{key}", d["G1"]["mean"]); put(f"DSixGHalf{key}", d["G05"]["mean"]); put(f"DSixShare{key}", d["share_G05_over_G1"])
    # D07
    put("DSevenSdTwentyFour", D["D07"]["cremad_24"]["_sd_over_draws"]); put("DSevenSdNinetyOne", D["D07"]["cremad_91m"]["_sd_over_draws"])
    v24 = [v for k, v in D["D07"]["cremad_24"].items() if not k.startswith("_")]; v91 = [v for k, v in D["D07"]["cremad_91m"].items() if not k.startswith("_")]
    put("DSevenMeanTwentyFour", float(np.mean(v24))); put("DSevenMeanNinetyOne", float(np.mean(v91)))
    # D08 / D09
    put("DEightRRminusGG", D["D08"]["rr_minus_gg"]["mean"]); put("DEightRRminusGGLo", D["D08"]["rr_minus_gg"]["ci_low"]); put("DEightRRminusGGHi", D["D08"]["rr_minus_gg"]["ci_high"])
    for cell, key in (("RR_hpo", "RR"), ("GR_hpo", "GR"), ("GG_hpo", "GG")):
        put(f"DEightOptimism{key}", D["D08"][cell]["test_selection_optimism_mean"])
    put("DNine", D["D09"]["mean"]); put("DNineLo", D["D09"]["ci_low"]); put("DNineHi", D["D09"]["ci_high"])
    # D14
    for lv, key in (("ravdess", "Rav"), ("subesco_980", "Sub")):
        put(f"DFourteenSdDraw{key}", D["D14"][lv]["sd_draw"]); put(f"DFourteenSdSeed{key}", D["D14"][lv]["sd_seed"]); put(f"DFourteenGrand{key}", D["D14"][lv]["grand_mean"])
    # D15 / D16 / D18
    p = a.run / "descriptors" / "d15_d16_d18.json"
    if p.exists():
        d = read_json(p)
        put("DFifteenLOSO", d["D15"]["LOSO"]["mean_speaker_uar"]); put("DFifteenLOSOSUB", d["D15"]["LOSOSUB"]["mean_speaker_uar"]); put("DFifteenGFive", d["D15"]["GG"]["mean_speaker_uar"])
        put("DFifteenLOSOminusGFive", d["D15"]["LOSO_minus_Group5"]["mean"]); put("DFifteenLOSOminusGFiveLo", d["D15"]["LOSO_minus_Group5"]["ci_low"]); put("DFifteenLOSOminusGFiveHi", d["D15"]["LOSO_minus_Group5"]["ci_high"])
        put("DFifteenSUBminusGFive", d["D15"]["LOSOSUB_minus_Group5"]["mean"]); put("DFifteenSUBminusGFiveLo", d["D15"]["LOSOSUB_minus_Group5"]["ci_low"]); put("DFifteenSUBminusGFiveHi", d["D15"]["LOSOSUB_minus_Group5"]["ci_high"])
        for c, key in (("ravdess", "Rav"), ("cremad", "Cre")):
            e = d["D16"][c]
            put(f"DSixteenPOneRR{key}", e["P1RR"]["uar_pooled_mean"]); put(f"DSixteenPOneGG{key}", e["P1GG"]["uar_pooled_mean"])
            put(f"DSixteenPremVTwo{key}", e["premium_v2_on_P1_splits"]); put(f"DSixteenPremPOne{key}", e["premium_p1_2026"])
            put(f"DSixteenPOneRowRR{key}", e["p1_2026"]["random"]); put(f"DSixteenPOneRowGG{key}", e["p1_2026"]["groupkfold"])
    # HPO on RAVDESS (secondary level): RR_hpo - GG_hpo per speaker from the cell-runs (descriptor, no inference)
    def hpo_cell(level, cell):
        for key, cr in R["cell_runs"].items():
            arm, lv, m, c, rr, s = key.split("|")
            if arm == "HPO" and lv == level and c == cell and cr["complete"] and rr[1:] == s[1:]:
                return cr["speaker_uar"]
        return None
    rr, gg = hpo_cell("ravdess", "RR_hpo"), hpo_cell("ravdess", "GG_hpo")
    if rr and gg:
        from ser_v2 import stats as _stats
        dd = np.asarray([rr[s] - gg[s] for s in sorted(gg) if s in rr])
        lo, hi = _stats.speaker_bootstrap_ci(dd)
        put("DEightRRminusGGRav", float(dd.mean())); put("DEightRRminusGGRavLo", lo); put("DEightRRminusGGRavHi", hi); put("DEightRavN", len(dd))
    # FT arm convergence (unit.json): frozen comparator best epoch == last epoch; fine-tune mean best epoch
    units_dir = a.run / "units"
    fr, ft = [], []
    for uj in units_dir.glob("*/unit.json"):
        j = read_json(uj)
        if j.get("arm") == "FT":
            (fr if j["model"].endswith("frozen_sr") else ft).append(j)
    if fr:
        put("DFtFrozenBestLast", sum(1 for j in fr if j["best_epoch"] == j["epochs_run"])); put("DFtFrozenTotal", len(fr))
        put("DFtFrozenMeanBest", float(np.mean([j["best_epoch"] for j in fr])), 1)
    if ft:
        put("DFtFtBestLast", sum(1 for j in ft if j["best_epoch"] == j["epochs_run"])); put("DFtFtTotal", len(ft)); put("DFtFtMeanBest", float(np.mean([j["best_epoch"] for j in ft])), 1)
    # descriptor reconciliation D04/D07/D08 (tools.reconcile_descriptors)
    p = a.run / "descriptors" / "reconcile_d04_d07_d08.json"
    if p.exists():
        d = read_json(p)
        put("DReconcileItems", d["n_items"]); put("DReconcileFailed", d["n_failed"])
        M["DReconcileMaxDiff"] = "--" if d["max_abs_diff"] is None else f"{d['max_abs_diff']:.1e}"
    # D12 / D13
    LEVEL_DISPLAY = {"ravdess": "RAVDESS", "cremad": "CREMA-D", "subesco_980": "SUBESCO-980", "subesco_full": "SUBESCO-full",
                     "sub_1400_one": "the 1,400-combination one-take SUBESCO panel", "sub_700_one": "the 700-combination one-take SUBESCO panel",
                     "sub_700_two": "the 700-combination two-take SUBESCO panel"}
    ENC_DISPLAY = {"hubert_base": "HuBERT-base", "wavlm_base_plus": "WavLM-base+", "wav2vec2_base": "wav2vec2-base"}
    p = a.run / "descriptors" / "ridge_sweep.json"
    if p.exists():
        d = read_json(p)
        st = d["D13"]; put("DThirteenStable", sum(1 for v in st.values() if v["stable"])); put("DThirteenTotal", len(st))
        put("DThirteenAllTwenty", sum(1 for v in st.values() if v["n_positive"] == v["n_draws"]))
        unstable = [k for k, v in st.items() if not v["stable"]]
        M["DThirteenUnstable"] = "; ".join(f"{LEVEL_DISPLAY.get(k.split('|')[0], k.split('|')[0])} with {ENC_DISPLAY.get(k.split('|')[1], k.split('|')[1])}" for k in unstable) or "none"
        rhos = [v["spearman_rho_state"] for v in d["D12"].values()]
        put("DTwelveNegRho", sum(1 for r in rhos if r < 0)); put("DTwelveTotal", len(rhos)); put("DTwelveMedianRho", float(np.median(rhos)))
        early = [max(v["mean_by_state"][1:4]) for v in d["D12"].values()]; late = [v["mean_by_state"][12] for v in d["D12"].values()]
        put("DTwelveEarlyOverLate", float(np.median(np.asarray(early) / np.maximum(np.asarray(late), 1e-9))), 1)
    # D10 / D11
    p = a.run / "descriptors" / "mechid.json"
    if p.exists():
        d = read_json(p)
        for key, tag in (("CTRL|ravdess|cnn", "CtrlRavCnn"), ("CTRL|cremad|cnn", "CtrlCreCnn"), ("CTRL|subesco_980|cnn", "CtrlSubCnn"),
                         ("CTRL|ravdess|resnet_se", "CtrlRavRes"), ("CTRL|cremad|resnet_se", "CtrlCreRes"), ("CTRL|subesco_980|resnet_se", "CtrlSubRes"),
                         ("FT|ravdess|wavlm_base_plus_ft", "FtRav"), ("FT|cremad|wavlm_base_plus_ft", "FtCre"), ("FT|subesco_980|wavlm_base_plus_ft", "FtSub"),
                         ("FT|ravdess|wavlm_base_plus_frozen_sr", "FrRav"), ("FT|cremad|wavlm_base_plus_frozen_sr", "FrCre"), ("FT|subesco_980|wavlm_base_plus_frozen_sr", "FrSub")):
            e = d["D10"].get(key, {})
            put(f"DTen{tag}", e.get("mean_RR_minus_GG_above_chance")); put(f"DTen{tag}RR", e.get("mean_RR_above_chance")); put(f"DTen{tag}GG", e.get("mean_GG_above_chance"))
        vals = [v["mean_RR_minus_GG_above_chance"] for k, v in d["D10"].items() if v.get("tested")]
        put("DTenMedianAll", float(np.median(vals)) if vals else None); put("DTenPositive", sum(1 for v in vals if v > 0)); put("DTenTotal", len(vals))
        for key, tag in (("CTRL|cremad|cnn", "CtrlCreCnn"), ("CTRL|ravdess|cnn", "CtrlRavCnn"), ("CTRL|subesco_980|cnn", "CtrlSubCnn"), ("FT|cremad|wavlm_base_plus_ft", "FtCre")):
            e = d["D11"].get(key, {})
            for cell in ("RR", "GG"):
                for cat in ("sibling", "same_speaker", "same_sentence", "other"):
                    put(f"DEleven{tag}{cell}{cat.replace('_', '').title()}", 100 * e.get(cell, {}).get(cat, float('nan')), 0)
    # STAT retrain check (runs/<run>/../retrain_check/retrain_report.json)
    p = a.run.parent / "retrain_check" / "retrain_report.json"
    if p.exists():
        d = read_json(p)
        put("DRetrainTotal", d["n_selected"]); put("DRetrainIdentical", d["identical"]); put("DRetrainDifferent", d["different"])
        diff = [u for u in d["units"] if u["status"] != "identical"]
        if diff:
            u = diff[0]
            put("DRetrainAgree", 100 * u["y_pred_agreement"], 1); put("DRetrainUarMain", u["uar_main"]); put("DRetrainUarCheck", u["uar_check"]); put("DRetrainMaxLogit", u["max_abs_logit_diff"])
    lines = ["% generated by tools.paper_descriptors from results.json and runs/main/descriptors/*.json; do not edit by hand"]
    for k, v in M.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(len(M), "macros ->", a.out)


if __name__ == "__main__":
    main()
