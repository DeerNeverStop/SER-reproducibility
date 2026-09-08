"""Additional read-only checks of A and B2 numeric CSV exports after core acceptance."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,gzip,json
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
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def main():
    V.cpu_guard(2);counts=Counter();began=datetime.now(timezone.utc).isoformat()
    work=REPO/"v3/deploy/work";results=REPO/"v3/deploy/results";ar=results/"A"
    passed=read(REPO/"v3/deploy/evidence/analysis_verification_all_frozen.json")
    for module in ("A","B2","B2-FIX"):need(passed["modules"][module]["comparison"]["pass"],"core module not accepted")
    plan=read(work/"plan/A.json");ae=read(ar/"endpoints.json")
    # class_order is the randomized reference sampling order; label_index is
    # defined by the original manifest and must determine raw-logit labels.
    labels_by_level={level:V.Manifest(REPO/"v2/manifests"/(block["base"]+"_manifest.csv")).classes for level,block in plan["levels"].items()}
    eps=ae["endpoints"];absolute=ae["absolute"]
    harm=[dict(level=x["level"],enc=x["model"],condition=x["condition"],estimator=x["estimator"],share=x["estimate"],ci_lo=x["ci95"][0],ci_hi=x["ci95"][1],n=x["n"]) for x in eps if x["endpoint"]==7 and x.get("available",True)]
    counts["A_harm_numeric"]=D.compare_records(harm,V.read_csv_rows(ar/"harm_share.csv"),("level","enc","condition","estimator"),"A harm export")
    # Reconstruct absolute class recall from raw records. This extra audit only
    # needs naive single-emotion and global baseline rows; all rows/artifacts
    # were already fully integrity checked by the frozen verifier.
    sums=defaultdict(lambda:[0,0])
    for u in plan["units"]:
        path=work/"A/units"/u["unit_id"]/"predictions.csv.gz"
        need(V.sha256_file(path)==(path.parent/"DONE").read_text().split()[0],"A raw prediction receipt changed")
        classes=labels_by_level[u["level"]]
        single={c["id"]:classes.index(c["composition"].split(":",1)[1]) for c in plan["levels"][u["level"]]["conditions"] if c["feasible"] and c["composition"].startswith("single:")}
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            reader=csv.reader(f);need(next(reader)==["condition","estimator","draw","relative_path","speaker","y_true","y_pred"],"A raw columns")
            for row in reader:
                condition,estimator=row[:2]
                if not ((condition=="none" and estimator=="none") or (condition in single and estimator=="naive")):continue
                truth=int(row[5])
                if condition in single and truth!=single[condition]:continue
                key=(u["level"],u["enc"],condition,u["r"],int(row[2]),row[4],truth)
                sums[key][0]+=int(row[6])==truth;sums[key][1]+=1
        counts["A_raw_files_for_absolute_recall"]+=1
    def recall_mean(level,encoder,condition,cls):
        speakers=plan["levels"][level]["speakers"]
        draws=[0] if condition=="none" else range(5)
        per_speaker=[]
        for s in speakers:
            repetitions=[]
            for rr in range(3):
                values=[]
                for draw in draws:
                    key=(level,encoder,condition,rr,draw,s,cls)
                    need(key in sums and sums[key][1]>0,"absolute recall support/repeat missing")
                    a,n=sums[key];values.append(100*a/n)
                repetitions.append(float(np.mean(values)))
            per_speaker.append(float(np.mean(repetitions)))
        return float(np.mean(per_speaker))
    rec=[]
    for e in eps:
        if e["endpoint"]!=4 or not e.get("available",True) or e.get("target_class") is None:continue
        classes=labels_by_level[e["level"]];c=classes.index(e["target_class"])
        row=dict(level=e["level"],enc=e["model"],N=e["N"],recall_single=recall_mean(e["level"],e["model"],e["condition"],c),recall_none=recall_mean(e["level"],e["model"],"none",c),diff=e["estimate"],ci_lo=e["ci95"][0],ci_hi=e["ci95"][1],n=e["n"])
        row["class"]=e["target_class"];rec.append(row)
        counts["A_recall_difference_identity"]+=D.assert_numeric(row["recall_single"]-row["recall_none"],row["diff"])
    counts["A_recall_export_numeric"]=D.compare_records(rec,V.read_csv_rows(ar/"recall.csv"),("level","enc","class","N"),"A recall export")
    def endpoint(level,model,number,condition,estimator="naive"):
        rows=[x for x in eps if x["level"]==level and x["model"]==model and x["endpoint"]==number and x["condition"]==condition and x["estimator"]==estimator and x.get("available",True)]
        need(len(rows)==1,"A summary endpoint mapping not unique")
        return rows[0]
    def absolute_row(level,model,condition,estimator="naive"):
        rows=[x for x in absolute if x["level"]==level and x["model"]==model and x["condition"]==condition and x["estimator"]==estimator]
        need(len(rows)==1,"A summary absolute mapping not unique")
        return rows[0]
    primary=[];n5=[]
    for level in plan["levels"]:
        for model in sorted({u["enc"] for u in plan["units"] if u["level"]==level}):
            baseline=absolute_row(level,model,"none","none")
            row=dict(level=level,model=model,n_speakers=baseline["n"],global_uar=baseline["estimate"],balanced_N3_naive_uar=absolute_row(level,model,"balanced@3")["estimate"])
            row["single_N3_naive_uar"]=baseline["estimate"]+endpoint(level,model,2,"single:*@3")["estimate"]
            fields=[("single_minus_global",2,"single:*@3","naive"),("single_minus_balanced",3,"single:*@3","naive"),("class_matched_secondary",3,"single:balanced_classes@3","naive"),("reference_class_recall_change",4,"single:*@3","naive"),("other_minus_self_balanced",6,"other_balanced@3","naive"),("other_minus_self_single",6,"other_single:*@3","naive")]
            fields += [("recovery_"+est,5,"single:*@3",est) for est in ("shrink","neutral_filter","prior_corrected")]
            fields += [("harm_share_"+est,7,"single:*@3",est) for est in ("naive","shrink","neutral_filter","prior_corrected")]
            for name,number,condition,estimator in fields:
                e=endpoint(level,model,number,condition,estimator);row.update({name:e["estimate"],name+"_lo":e["ci95"][0],name+"_hi":e["ci95"][1]})
            primary.append(row)
            feasible=any(c["feasible"] and c["N"]==5 and c["composition"].startswith("single:") for c in plan["levels"][level]["conditions"])
            row=dict(level=level,model=model,balanced_N5_naive_uar=absolute_row(level,model,"balanced@5")["estimate"],single_N5_feasible=str(feasible))
            if feasible:
                for name,number in (("single_minus_global",2),("single_minus_balanced",3)):
                    e=endpoint(level,model,number,"single:*@5");row.update({name:e["estimate"],name+"_lo":e["ci95"][0],name+"_hi":e["ci95"][1]})
                row["naive_harm_share"]=endpoint(level,model,7,"single:*@5")["estimate"]
            else:
                for name in ("single_minus_global","single_minus_global_lo","single_minus_global_hi","single_minus_balanced","single_minus_balanced_lo","single_minus_balanced_hi","naive_harm_share"):row[name]=""
            n5.append(row)
    for filename,expected in (("primary_N3.csv",primary),("primary_all9.csv",primary),("N5.csv",n5),("secondary_N5_all9.csv",n5)):
        counts["A_"+filename+"_numeric"]=D.compare_records(expected,V.read_csv_rows(ar/filename),("level","model"),"A summary "+filename)
    # B2's row export is recomputed for each of the 390 raw val/test pairs,
    # ensuring a swapped threshold cannot hide in unchanged aggregate means.
    bp=read(work/"plan/B2.json");unit_expected=[]
    for u in bp["units"]:
        path=V.b2_unit_path(work,u["unit_id"],u["model"])
        pred={role:V.read_logit_predictions(path/(role+"_predictions.csv"),False) for role in ("val","test")}
        result=D.calibration_unit(pred["val"],pred["test"])
        row={k:u[k] for k in ("unit_id","level","model","cell","r","fold")}
        for rule in ("primary","secondary"):
            for key,value in result[rule].items():row[rule+"."+key]=value
        unit_expected.append(row)
    actual=V.read_csv_rows(results/"B2/units.csv")
    for row in actual:
        for key in ("secondary.feasible",):
            if key in row:
                need(row[key] in ("True","False"),"boolean export value invalid");row[key]=row[key]=="True"
    counts["B2_unit_export_numeric"]=D.compare_records(unit_expected,actual,("unit_id",),"B2 unit export")
    files=[ar/x for x in ("endpoints.json","harm_share.csv","recall.csv","primary_N3.csv","primary_all9.csv","N5.csv","secondary_N5_all9.csv")]+[results/"B2/units.csv"]
    report=dict(program=V.PROGRAM,kind="independent-extra-export-verification",pass_=True,started_at=began,finished_at=datetime.now(timezone.utc).isoformat(),checks=dict(counts),
                compared=sum(v for k,v in counts.items() if k!="A_raw_files_for_absolute_recall"),source_script_path=str(Path(__file__).resolve()),source_script_sha256=V.sha256_file(Path(__file__)),
                artifact_sha256={str(p.relative_to(REPO)):V.sha256_file(p) for p in files},
                method="A absolute class recalls reconstructed from raw draw/repeat/speaker counts; A presentation exports mapped to already independently accepted endpoints; B2 unit CSV recalculated separately for every raw val/test pair. Frozen sources and raw outputs unmodified.")
    report["pass"]=report.pop("pass_")
    out=REPO/"v3/deploy/evidence/analysis_verification_extra_exports.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("pass","compared","checks")}))

if __name__=="__main__":main()
