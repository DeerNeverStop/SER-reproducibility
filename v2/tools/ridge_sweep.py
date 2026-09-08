"""PROBECPU sweep (descriptors D12 and D13; estimation only, never tested).

Ridge A1 head (StandardScaler on the training fold + RidgeClassifier alpha=1, balanced) on the
13 cached SSL states of the three encoders, under Random (StratifiedKFold) and Grouped
(StratifiedGroupKFold by speaker) five-fold outer partitions without inner split, for 20
partition draws per level (draw seed = seeds.draw_seed(level, d)).

Levels: the three family levels, subesco_full, and the three SUBESCO panels (7 levels).
Per (level, encoder, state, draw): premium = mean over speakers of [U_random(s) - U_grouped(s)]
where U is per-speaker UAR over the OOF predictions (same definition as the scorer).

D12  premium by state (0..12) per encoder x level, mean and SD over draws, Spearman rho of the
     draw-mean premium against the state index.
D13  direction stability at the contract state 12: positive draws out of 20 and the empirical
     5th percentile; the preregistered reading is ">= 18/20 positive and q05 > 0".

    python -m tools.ridge_sweep --plan plan_rc2 --manifests manifests --features features --out runs/main/descriptors/ridge_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))

from ser_v2 import corpora, seeds, splits, stats  # noqa: E402
from ser_v2.common import atomic_write_json, read_json  # noqa: E402
from ser_v2.features import FeatureStore  # noqa: E402
from ser_v2.run_plan import level_subset  # noqa: E402

ENCODERS = ("hubert_base", "wavlm_base_plus", "wav2vec2_base")
LEVELS = ("ravdess", "cremad", "subesco_980", "subesco_full", "sub_1400_one", "sub_700_one", "sub_700_two")
N_DRAWS = 20
N_STATES = 13


def ridge_oof(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    pred = np.full(len(y), -1)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred


def per_speaker_uar(y, pred, spk, K):
    out = {}
    for s in sorted(set(spk.tolist())):
        m = spk == s
        out[s] = stats.uar(y[m], pred[m], K)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    a = ap.parse_args(argv)
    feats = FeatureStore(a.features)
    pops, arrays = {}, {}
    for base in ("ravdess", "cremad", "subesco", "subesco_980"):
        rows, _ = corpora.apply_hygiene("subesco" if base == "subesco_980" else base, corpora.load_manifest(a.manifests / f"{base}_manifest.csv"))
        pops[base] = rows
        arrays[base] = corpora.manifest_arrays(rows)
    pinned = None
    out = {"D12": {}, "D13": {}, "n_draws": a.draws, "definition": __doc__.strip().splitlines()[0:16]}
    t0 = time.time()
    for level in a.levels.split(","):
        base = "subesco_980" if level == "subesco_980" else ("subesco" if level.startswith("sub") else level)
        A = arrays[base]
        sub = np.arange(len(A["y"]), dtype=np.int64) if base == "subesco_980" else level_subset(level, A, pops[base], 0, pinned)
        paths = [pops[base][i]["relative_path"] for i in sub]
        y, spk = A["y"][sub], A["speaker"][sub]
        K = len(corpora.CORPORA["subesco" if base.startswith("subesco") else base].labels)
        for enc in ENCODERS:
            Xall = feats.get(base, enc, paths)            # [n, 13, 768]
            prem = np.zeros((a.draws, N_STATES))
            for d in range(a.draws):
                sd = seeds.draw_seed(level, d)
                fr = splits.outer_random(y, sd)
                fg = splits.outer_grouped(y, spk, sd)
                for st in range(N_STATES):
                    X = Xall[:, st, :]
                    ur = per_speaker_uar(y, ridge_oof(X, y, fr), spk, K)
                    ug = per_speaker_uar(y, ridge_oof(X, y, fg), spk, K)
                    prem[d, st] = float(np.mean([ur[s] - ug[s] for s in ur]))
                print(f"{level} {enc} draw {d}: state12 premium {prem[d, 12]:.2f} ({time.time() - t0:.0f}s)", flush=True)
            from scipy.stats import spearmanr
            mean_by_state = prem.mean(axis=0)
            rho = spearmanr(np.arange(N_STATES), mean_by_state).statistic
            out["D12"][f"{level}|{enc}"] = {"mean_by_state": mean_by_state.tolist(), "sd_by_state": prem.std(axis=0, ddof=1).tolist(),
                                            "spearman_rho_state": float(rho), "premium_matrix_draw_by_state": prem.tolist()}
            p12 = prem[:, 12]
            out["D13"][f"{level}|{enc}"] = {"n_positive": int((p12 > 0).sum()), "n_draws": int(len(p12)), "mean": float(p12.mean()),
                                            "q05": float(np.percentile(p12, 5)), "q95": float(np.percentile(p12, 95)),
                                            "stable": bool((p12 > 0).sum() >= 18 and np.percentile(p12, 5) > 0)}
        atomic_write_json(a.out, out)
    atomic_write_json(a.out, out)
    print(json.dumps(out["D13"], indent=1))


if __name__ == "__main__":
    main()
