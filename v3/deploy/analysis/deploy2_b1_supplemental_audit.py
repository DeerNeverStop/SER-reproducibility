"""Post-result, explicitly versioned independent B1 acceptance supplement.

Frozen verifier and raw/scorer files remain unchanged. Two disclosed defects:
the frozen CSV exporter omits higher class support columns and the frozen
verifier's per-speaker nanmean conflicts with strict repeat risk. This audit
checks every other frozen expectation unchanged, then independently implements
strict repeat risk for ALL speakers/budgets/modes, including defined-repeat
counts. It additionally verifies all 72 per-repeat JSON/CSV exports from raw.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=next(candidate for parent in (HERE,*HERE.parents) for candidate in (parent,parent/"SER-deploy") if (candidate/"v3/deploy/EXECUTION_LOCK.json").is_file())
sys.path.insert(0,str(REPO))
import numpy as np
from v3.deploy import verify as V
from v3.deploy import verify_deploy2 as D

def need(ok,msg):
    if not ok:raise ValueError(msg)

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def strict_speaker_records(groups, by_fold):
    """Rule fixed for all data: any undefined repeat -> undefined speaker risk."""
    matrix=defaultdict(list)
    for (level,model,rr,seed),group in sorted(groups.items()):
        people=np.asarray(group["speakers"]); paths=np.asarray(group["paths"])
        errors=group["y_true"] != group["y_pred"]
        for budget in (.1,.2,.3):
            accepted=np.zeros(len(paths),bool)
            masks=[group["fold"]==f for f in sorted(set(group["fold"]))] if by_fold else [np.ones(len(paths),bool)]
            for mask in masks:accepted[mask]=D.quota(group["conf"][mask],paths[mask].tolist(),budget)
            for speaker in sorted(set(people)):
                mask=people==speaker;count=int((accepted&mask).sum())
                risk=float((errors&accepted&mask).sum()/count) if count else float("nan")
                matrix[(level,model,budget,speaker)].append((risk,int(mask.sum()),float(accepted[mask].mean())))
    result=[]
    for (level,model,budget,speaker),rows in sorted(matrix.items()):
        risks=np.asarray([r[0] for r in rows]); n_defined=int(np.isfinite(risks).sum())
        result.append(dict(level=level,model=model,b=budget,speaker=speaker,accepted_error=float(risks.mean()),
                           coverage=float(np.mean([r[2] for r in rows])),n_utts=rows[0][1],n_reps=len(rows),n_defined_risk_reps=n_defined))
    return result

def ranking(group,by_fold):
    errors=group["y_true"] != group["y_pred"]
    if not by_fold:return D.b1_ranking(group["conf"],errors,group["paths"])
    out=dict.fromkeys(("aurc","oracle_aurc","e_aurc"),0.)
    for fold in sorted(set(group["fold"])):
        mask=group["fold"]==fold
        row=D.b1_ranking(group["conf"][mask],errors[mask],np.asarray(group["paths"])[mask].tolist())
        for key in out:out[key]+=float(mask.mean())*row[key]
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-sidecar",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=Path("E:/科研/essay/SER-v2/v2"))
    parser.add_argument("--out",type=Path,default=REPO/"v3/deploy/evidence/analysis_verification_B1_supplemental.json")
    args=parser.parse_args()
    V.cpu_guard(2)
    began=datetime.now(timezone.utc).isoformat();checks=Counter();issues=[]
    results=REPO/"v3/deploy/results/B1";work=REPO/"v3/deploy/work/B1_analysis"
    frozen_report=REPO/"v3/deploy/evidence/analysis_verification_B1_frozen.json"
    need(frozen_report.is_file() and read(frozen_report)["pass"] is False,"original frozen failure must remain present")
    lock=read(REPO/"v3/deploy/EXECUTION_LOCK.json")
    for rel,sha in lock["files"].items():need(V.sha256_file(REPO/rel)==sha,"frozen source drift: "+rel)
    ctx=V.Context(V.Sources.real(args.data_root));rep=V.Report()
    groups=V.verify_b1_integrity(ctx,rep)
    need(not rep.failed(),"raw integrity failed: "+str(rep.failures))
    need(not rep.notes,"raw integrity notes require explicit review: "+str(rep.notes))
    need(len(groups)==72 and len({k[:2] for k in groups})==21,"B1 source group count changed")
    selected=V.select_b1_rows(ctx.src.run_plan,ctx.src.level_base)
    need(len(selected)==len({u['unit_id'] for u in selected})==360,"B1 source unit count changed")
    actual=read(results/"endpoints.json")
    need(actual["program"]==V.PROGRAM and actual["module"]=="B1","result identity")
    inputs=actual["inputs"]
    need(inputs["n_units"]==360 and inputs["v2_run_plan_sha256"]==V.sha256_file(ctx.src.run_plan),"source plan receipt")
    source_map={x["unit_id"]:x for x in inputs["units"]}
    need(len(source_map)==360 and set(source_map)=={u['unit_id'] for u in selected},"source receipt unit inventory")
    expected_replicates=defaultdict(list)
    for u in selected:
        uid=u["unit_id"];raw=args.data_root/"runs/main/units"/uid/"predictions.csv"
        need(source_map[uid]["predictions_sha256"]==V.sha256_file(raw),"source receipt raw SHA")
        key=f"{u['corpus_level']}|{u['model']}|r{u['r']}s{u['seed_index']}"
        expected_replicates[key].append(uid)
    need({k:sorted(v) for k,v in inputs["replicates"].items()}=={k:sorted(v) for k,v in expected_replicates.items()},"source replicate inventory")
    checks["source_units_SHA_identity"]=360
    files={"frozen_failure":frozen_report,"scorer_endpoints":results/"endpoints.json","scorer_original_summary":results/"summary.csv","complete_summary_sidecar":args.summary_sidecar,"scorer_per_speaker":results/"per_speaker.csv","scorer_class_mix":results/"rejected_class_mix.csv"}
    differences=[];mode_counts={};expectations={}
    for by_fold in (False,True):
        name="fold" if by_fold else "global"
        expected=D.b1_aggregate(groups,ctx,by_fold)
        expectations[name]=expected
        if by_fold:
            targets={key:[row for group in actual["fold_sensitivity"] for row in group[key]] for key in ("summary","per_speaker","mix","endpoints")}
        else:
            targets={"endpoints":actual["endpoints"],"summary":V.read_csv_rows(args.summary_sidecar),
                     "per_speaker":V.read_csv_rows(results/"per_speaker.csv"),"mix":V.read_csv_rows(results/"rejected_class_mix.csv")}
        for key,fields in (("endpoints",("level","model","quantity")),("summary",("level","model","b")),("mix",("level","model","b","class_index"))):
            checks[name+"_"+key+"_numeric"]=D.compare_records(expected[key],targets[key],fields,"supplemental B1 "+name+" "+key)
        # Frozen finite-only expected risk is retained for explicit diagnostics;
        # it is NOT silently relabelled as the strict expectation.
        strict=strict_speaker_records(groups,by_fold)
        original={tuple(row[k] for k in ("level","model","b","speaker")):row for row in expected["per_speaker"]}
        for row in strict:
            key=tuple(row[k] for k in ("level","model","b","speaker"));old=original[key]
            if np.isfinite(old["accepted_error"]) != np.isfinite(row["accepted_error"]):
                differences.append(dict(mode=name,level=row["level"],model=row["model"],b=row["b"],speaker=row["speaker"],n_reps=row["n_reps"],n_defined_risk_reps=row["n_defined_risk_reps"],frozen_expected_is_finite=bool(np.isfinite(old["accepted_error"])),strict_expected_is_finite=bool(np.isfinite(row["accepted_error"]))))
        checks[name+"_strict_per_speaker_numeric"]=D.compare_records(strict,targets["per_speaker"],("level","model","b","speaker"),"strict B1 "+name+" per speaker")
        mode_counts[name]={key:len(value) for key,value in targets.items()}
    expected_json_names=set();expected_csv_names=set();flat_metrics=[];flat_speakers=[]
    for (level,model,rr,seed),group in sorted(groups.items()):
        need(group["C"]==ctx.manifest(level).n_classes,"raw logit width against fixed classes")
        base=f"{level}__{model}__r{rr}s{seed}";jpath=work/(base+".json");cpath=work/(base+".csv")
        expected_json_names.add(jpath.name);expected_csv_names.add(cpath.name)
        saved=read(jpath);checks["replicate_header_numeric"]+=D.assert_numeric({"n_speakers":len(set(group["speakers"])),"n_utts":len(group["paths"])},saved)
        csv_rows=[]
        for by_fold in (False,True):
            key="fold_budgets" if by_fold else "budgets";rank_key="fold_aurc" if by_fold else "aurc"
            checks["replicate_ranking_numeric"]+=D.assert_numeric(ranking(group,by_fold),saved[rank_key])
            need(set(saved[key])=={"0.1","0.2","0.3"},"replicate budgets changed")
            for budget in (.1,.2,.3):
                expected=D.b1_budget(group,budget,by_fold)
                checks["replicate_budget_numeric"]+=D.assert_numeric(expected,saved[key][str(budget)])
                variant="within_fold" if by_fold else "global_oof"
                flat_metrics.append(dict(level=level,model=model,replicate=f"r{rr}s{seed}",variant=variant,**expected["metrics"]))
                for speaker,row in expected["speakers"].items():
                    flat_speakers.append(dict(level=level,model=model,replicate=f"r{rr}s{seed}",variant=variant,b=budget,speaker=speaker,**row))
                if not by_fold:
                    for speaker,row in expected["speakers"].items():
                        csv_rows.append(dict(level=level,model=model,rep=f"r{rr}s{seed}",b=budget,speaker=speaker,**row))
        checks["replicate_speaker_csv_numeric"]+=D.compare_records(csv_rows,V.read_csv_rows(cpath),("level","model","rep","b","speaker"),"replicate CSV")
        files[base+".json"]=jpath;files[base+".csv"]=cpath
    need({p.name for p in work.glob('*.json')}==expected_json_names|{"source_inventory.json"},"replicate JSON inventory plus separately identified source manifest")
    files["source_inventory.json"]=work/"source_inventory.json"
    need({p.name for p in work.glob('*.csv')}==expected_csv_names,"replicate CSV inventory")
    for name,expected,fields in (("fold_sensitivity_summary.csv",expectations["fold"]["summary"],("level","model","b")),
                                  ("per_repeat_metrics_complete.csv",flat_metrics,("level","model","replicate","variant","b")),
                                  ("per_repeat_speakers_complete.csv",flat_speakers,("level","model","replicate","variant","b","speaker"))):
        path=results/name;files[name]=path
        checks["sidecar_"+name+"_numeric"]=D.compare_records(expected,V.read_csv_rows(path),fields,"sidecar "+name)
    # Independently validate the publication-facing primary table, including
    # its pooled versus equal-speaker and global versus fold distinctions.
    by_fold={(r["level"],r["model"],r["b"]):r for r in expectations["fold"]["summary"]}
    eps={(r["level"],r["model"],r["quantity"]):r for r in expectations["global"]["endpoints"]}
    primary=[]
    for row in expectations["global"]["summary"]:
        if row["b"]!=.2:continue
        level,model=row["level"],row["model"];f=by_fold[(level,model,.2)]
        cov=[r["coverage"] for r in expectations["global"]["per_speaker"] if r["level"]==level and r["model"]==model and r["b"]==.2]
        primary.append(dict(row,speaker_equal_coverage=float(np.mean(cov)),fold_accepted_error=f["accepted_error"],
                            fold_actual_review_rate=f["actual_review_rate"],fold_speakers_below_0_70=f["speakers_below_0.70"],
                            accepted_error_ci_low=eps[(level,model,"accepted_error")]["ci95"][0],accepted_error_ci_high=eps[(level,model,"accepted_error")]["ci95"][1],
                            capture_ci_low=eps[(level,model,"error_capture_rate")]["ci95"][0],capture_ci_high=eps[(level,model,"error_capture_rate")]["ci95"][1]))
    path=results/"primary_budget20_complete.csv";files[path.name]=path
    checks["primary_sidecar_numeric"]=D.compare_records(primary,V.read_csv_rows(path),("level","model","b"),"primary sidecar")
    provenance=read(results/"complete_export_provenance.json");files["complete_export_provenance.json"]=results/"complete_export_provenance.json"
    need(provenance["original_summary_sha256"]==V.sha256_file(results/"summary.csv") and provenance["original_endpoints_sha256"]==V.sha256_file(results/"endpoints.json"),"sidecar original output provenance")
    need(provenance["frozen_scorer_sha256"]==lock["files"]["v3/deploy/review_budget.py"],"sidecar frozen scorer provenance")
    need(len(provenance["source_replicate_json_sha256"])==72,"sidecar source receipt count")
    for rel,value in provenance["source_replicate_json_sha256"].items():need(V.sha256_file(REPO/rel)==value,"sidecar source provenance")
    # Explicit rule test: a zero-acceptance repetition cannot disappear.
    need(np.isnan(np.mean([.4,float('nan')])) and np.isfinite(np.nanmean([.4,float('nan')])),'strict-vs-legacy counterexample')
    report={"schema":"ser26-deploy2-B1-post-result-supplemental-verification-1","program":V.PROGRAM,"pass":True,"frozen_verifier_pass":False,
            "started_at":began,"finished_at":datetime.now(timezone.utc).isoformat(),"lock_sha256":lock["lock_sha256"],"checks":dict(checks),"compared":sum(checks.values()),
            "source_units":360,"OOF_replicates":72,"level_model_groups":21,"table_rows":mode_counts,"strict_rule_disagreements_with_frozen_verifier":differences,
            "raw_or_frozen_sources_changed":False,"metric_definitions_or_budgets_changed":False,
            "frozen_failure_reasons":read(frozen_report)["reasons"],"source_script_path":str(Path(__file__).resolve()),"source_script_sha256":V.sha256_file(Path(__file__)),
            "artifact_sha256":{key:V.sha256_file(path) for key,path in files.items()},
            "limitations":["This supplemental acceptance was written after observing the frozen verifier failure and is disclosed as such. It does not turn the original frozen verification into a pass.","All frozen statistical rules were checked unchanged except the frozen verifier's finite-only speaker repeat risk, which is independently replaced here by the specification/scorer's strict ALL-repeat rule for every speaker, budget and mode. Original finite-only disagreements are reported explicitly.","The original summary.csv remains incomplete for higher class support columns. The complete sidecar is verified against raw predictions; the original file is not rewritten.","Intervals condition on existing OOF predictions and masks, use shared full-speaker bootstrap indices, and do not include re-fitting/calibration uncertainty or establish real deployment performance."]}
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("pass","frozen_verifier_pass","compared","table_rows","strict_rule_disagreements_with_frozen_verifier")},ensure_ascii=False))

if __name__=="__main__":main()
