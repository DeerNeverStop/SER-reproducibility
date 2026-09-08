"""LaTeX tables and a compact JSON digest for the v2 manuscript, generated from the scored run
(results.json) and the frozen split tables. Every number in the tables comes from these files;
nothing is typed by hand.

Outputs (in --out dir):
  tables_v2.tex     \\newcommand tables: \\TabFactorial (per level x model: RR, RG, GR, GG UAR and the three
                    contrasts), \\TabMech (mechanism block incl. exposed / unexposed half-exposure values read per
                    speaker from the split-table meta, exactly as the scorer's T02/D06 do), \\TabFT (WavLM FT vs
                    frozen), \\TabHypotheses (registry verdict table)
  digest_v2.json    the same numbers as JSON

    python -m tools.paper_tables --plan plan_rc2 --registry registry --run runs/main --out paper_assets
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
from ser_v2.common import atomic_write_json, atomic_write_text, read_csv, read_json  # noqa: E402

LEVEL_NAME = {"ravdess": "RAVDESS", "cremad": "CREMA-D", "subesco_980": "SUBESCO-980", "subesco_full": "SUBESCO-full"}
GROUPS = {"scratch": ("cnn", "resnet_se", "transformer"), "probe": ("hubert_base", "wavlm_base_plus", "wav2vec2_base")}
MODEL_NAME = {"cnn": "CNN", "resnet_se": "ResNet-SE", "transformer": "Transformer", "hubert_base": "HuBERT probe",
              "wavlm_base_plus": "WavLM+ probe", "wav2vec2_base": "wav2vec2 probe", "wavlm_base_plus_ft": "partial FT",
              "wavlm_base_plus_frozen_sr": "frozen, same regime"}


def cell_runs(results, arm, level, model, cell, r=None):
    out = []
    for key, cr in results["cell_runs"].items():
        a, lv, m, c, rr, s = key.split("|")
        if a != arm or lv != level or m != model or c != cell or not cr["complete"]:
            continue
        if arm in ("CTRL", "MECH2X2", "HPO") and rr[1:] != s[1:]:
            continue
        if r is not None and int(rr[1:]) != r:
            continue
        out.append((int(rr[1:]), cr["speaker_uar"]))
    return out


def cell_mean(results: dict, arm: str, level: str, model: str, cell: str) -> tuple[float | None, int]:
    """Mean over speakers of the replicate-averaged per-speaker UAR (main replicates only)."""
    acc = defaultdict(list)
    for _, su in cell_runs(results, arm, level, model, cell):
        for spk, v in su.items():
            acc[spk].append(v)
    if not acc:
        return None, 0
    return float(np.mean([np.mean(v) for v in acc.values()])), len(next(iter(acc.values())))


def half_exposure_values(results: dict, plan_dir: Path, plan_rows: list[dict], level: str, model: str):
    """Absolute UAR of the half-exposure conditions read per speaker: for each replicate r, a speaker's
    exposed value is its UAR in the h1/h2 cell where it is in meta.exposed_speakers of the spk_half_h1
    table (else h2), and its unexposed value is the other cell; averaged over replicates, then speakers.
    Mirrors ser_v2.score.Scorer.mech_half."""
    index = read_json(plan_dir / "split_index.json")
    sha_to_key = {v["sha256"]: k for k, v in index.items()}
    exposed_by_r = {}
    for row in plan_rows:
        if row["arm"] == "MECH2X2" and row["corpus_level"] == level and row["model"] == model and row["cell"] == "spk_half_h1" and row["seed_index"] == row["r"]:
            r = int(row["r"])
            if r in exposed_by_r:
                continue
            table = read_json(plan_dir / "splits" / f"{sha_to_key[row['split_sha256']]}.json")
            ex = set()
            for f in table["folds"]:
                ex.update(f["meta"].get("exposed_speakers", []))
            exposed_by_r[r] = ex
    per_r_exp, per_r_unexp = {}, {}
    for r, ex in sorted(exposed_by_r.items()):
        h1 = dict(cell_runs(results, "MECH2X2", level, model, "spk_half_h1", r)); h2 = dict(cell_runs(results, "MECH2X2", level, model, "spk_half_h2", r))
        if r not in h1 or r not in h2:
            continue
        h1, h2 = h1[r], h2[r]
        e, u = {}, {}
        for s in sorted(set(h1) & set(h2)):
            if s in ex:
                e[s], u[s] = h1[s], h2[s]
            else:
                e[s], u[s] = h2[s], h1[s]
        per_r_exp[r], per_r_unexp[r] = e, u
    if not per_r_exp:
        return None, None
    spk = sorted(set.intersection(*[set(v) for v in per_r_exp.values()]))
    exp = float(np.mean([np.mean([per_r_exp[r][s] for r in per_r_exp]) for s in spk]))
    une = float(np.mean([np.mean([per_r_unexp[r][s] for r in per_r_unexp]) for s in spk]))
    return exp, une


def f(x, nd=2):
    return "--" if x is None else (f"$-${abs(x):.{nd}f}" if x < 0 else f"{x:.{nd}f}")


def ci(h):
    if not h or not h.get("tested"):
        return "--"
    return f"{f(h['mean'])} [{f(h['ci_low'])}, {f(h['ci_high'])}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    R = read_json(a.run / "results.json")
    plan_rows = read_csv(a.plan / "run_plan.csv")
    reg = {h["hypothesis_id"]: h for h in read_csv(a.registry / "hypothesis_registry.csv")}
    H = {**R["hypotheses"], **R["tost"]}
    digest = {"factorial": {}, "mech": {}, "ft": {}, "hypotheses": {}, "claims": R["claims"]}
    lines = []
    # ---- factorial table
    lines.append("\\newcommand{\\TabFactorial}{%")
    lines.append("\\begin{tabular}{llrrrrrrr}\\toprule")
    lines.append("Corpus & Model & RR & RG & GR & GG & RR$-$GG & RG$-$GG & RR$-$RG \\\\\\midrule")
    for level in ("ravdess", "cremad", "subesco_980"):
        for grp, models in GROUPS.items():
            for m in models:
                vals = {c: cell_mean(R, "CTRL", level, m, c)[0] for c in ("RR", "RG", "GR", "GG")}
                digest["factorial"][f"{level}|{m}"] = vals
                d = lambda x, y: None if vals[x] is None or vals[y] is None else vals[x] - vals[y]
                lines.append(f"{LEVEL_NAME[level]} & {MODEL_NAME[m]} & {f(vals['RR'])} & {f(vals['RG'])} & {f(vals['GR'])} & {f(vals['GG'])} & {f(d('RR','GG'))} & {f(d('RG','GG'))} & {f(d('RR','RG'))} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}}")
    # ---- mechanism table
    lines.append("\\newcommand{\\TabMech}{%")
    lines.append("\\begin{tabular}{llrrrrrrr}\\toprule")
    lines.append("Corpus & Model & none & spk & prm & both & both+sib & exposed half & unexposed half \\\\\\midrule")
    for level in ("cremad", "subesco_full"):
        for m in ("cnn", "resnet_se"):
            vals = {c: cell_mean(R, "MECH2X2", level, m, c)[0] for c in ("none", "spk", "prm", "both", "both_sib")}
            exp, une = half_exposure_values(R, a.plan, plan_rows, level, m)
            vals["exposed_half"], vals["unexposed_half"] = exp, une
            digest["mech"][f"{level}|{m}"] = vals
            lines.append(f"{LEVEL_NAME[level]} & {MODEL_NAME[m]} & {f(vals['none'])} & {f(vals['spk'])} & {f(vals['prm'])} & {f(vals['both'])} & {f(vals['both_sib'])} & {f(exp)} & {f(une)} \\\\")
    lines.append("\\bottomrule\\end{tabular}}")
    # ---- FT table
    lines.append("\\newcommand{\\TabFT}{%")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{llrrr}\\toprule")
    lines.append("Corpus & Model & RR & GG & RR$-$GG \\\\\\midrule")
    for level in ("ravdess", "cremad", "subesco_980"):
        for m in ("wavlm_base_plus_ft", "wavlm_base_plus_frozen_sr"):
            rr, gg = cell_mean(R, "FT", level, m, "RR")[0], cell_mean(R, "FT", level, m, "GG")[0]
            digest["ft"][f"{level}|{m}"] = {"RR": rr, "GG": gg}
            lines.append(f"{LEVEL_NAME[level]} & {MODEL_NAME[m]} & {f(rr)} & {f(gg)} & {f(None if rr is None or gg is None else rr - gg)} \\\\")
    lines.append("\\bottomrule\\end{tabular}}")
    # ---- hypothesis table
    lines.append("\\newcommand{\\TabHypotheses}{%")
    lines.append("\\begin{tabular}{llllrrrl}\\toprule")
    lines.append("ID & Claim & Family & Contrast & $n$ & Mean [95\\% CI] & $p_{\\mathrm{Holm}}$ & Verdict \\\\\\midrule")
    for hid in sorted(H):
        h, r = H[hid], reg[hid]
        p = h.get("p_holm")
        ptxt = "--" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
        contrast = r["contrast"].replace("_", "\\_").replace("[", "{[}").replace("]", "{]}")
        lines.append(f"{hid} & {r['claim_id']} & {r['family']} & {contrast} & {h.get('n', 0)} & {ci(h)} & {ptxt} & {h['verdict']} \\\\")
        digest["hypotheses"][hid] = {"claim": r["claim_id"], "family": r["family"], "n": h.get("n"), "mean": h.get("mean"), "ci": [h.get("ci_low"), h.get("ci_high")],
                                     "p": h.get("wilcoxon_p", h.get("p")), "p_holm": p, "verdict": h["verdict"], "status": r["a_priori_status"]}
    lines.append("\\bottomrule\\end{tabular}}")
    a.out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(a.out / "tables_v2.tex", "% generated by tools.paper_tables from results.json and the split tables; do not edit by hand\n" + "\n".join(lines) + "\n")
    atomic_write_json(a.out / "digest_v2.json", digest)
    print(json.dumps(digest["mech"], indent=1))


if __name__ == "__main__":
    main()
