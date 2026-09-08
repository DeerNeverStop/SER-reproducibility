"""MECHID (descriptors D10 and D11; estimation only, never tested).

Runs AFTER the CTRL and FT arms, on the checkpoints the runner persisted
(`runs/<run>/units/<unit_id>/checkpoint.pt`: CTRL r=0/seed 0 cnn and resnet_se in cells RR and GG
on the family levels; every FT unit).  Nothing here feeds a hypothesis.

D10  speaker-ID probe per checkpoint on a common held-out panel.
     For outer fold f, S_f = the test speakers of the GG table of that level (r=0 for scratch;
     the FT tables reuse the r=0 CTRL partitions).  The RR checkpoint of fold f is probed on the
     RR test items of fold f whose speaker is in S_f; the GG checkpoint of fold f on its own test
     items (all of S_f).  Per speaker the two panels are trimmed to the same item count (hash
     order), so both panels share speaker set and size, and neither panel was used to train its
     checkpoint.  Probe = multinomial logistic regression (C=1, standardized embeddings),
     stratified 5-fold CV inside the panel, metric = macro speaker recall (%) minus chance
     100/|S_f|.  Reported: per fold, and the mean over folds of RR minus GG, per level x model.
     Embeddings: the pooled representation right before the classification head (CNN:
     features(x).mean(-1); ResNet-SE: blocks(lift(x)).mean(-1); WavLM: last-state time mean).
D11  nearest-neighbour composition: for every test item of a checkpoint, the cosine-nearest
     TRAINING item (fit + val of that fold) in the same embedding space, categorised as
     sibling (same speaker, sentence and label), same_speaker, same_sentence (different speaker),
     or other; proportions per level x model x cell.

    python -m tools.mechid --plan plan_rc2 --manifests manifests --features features --run runs/main
                           --audio-root-ravdess ... --audio-root-cremad ... --audio-root-subesco ...
                           --out runs/main/descriptors/mechid.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2.parent))

from ser_v2.common import atomic_write_json, read_csv, read_json  # noqa: E402
from ser_v2.features import FeatureStore  # noqa: E402
from ser_v2.train import PlanIO  # noqa: E402

FAMILY = ("ravdess", "cremad", "subesco_980")


def _hash_order(items: list[str], salt: str) -> list[str]:
    return sorted(items, key=lambda p: hashlib.sha256(f"{salt}|{p}".encode()).hexdigest())


# ----------------------------------------------------------------------------- embeddings

def scratch_embedder(model_name: str, cfg: dict, n_mels: int, n_classes: int, state_path: Path, device):
    import torch
    sys.path.insert(0, str(V2.parent))
    import advanced_models
    import tuned_standard_experiment as tse
    arch = dict(tse.candidate_configs(model_name)[2])
    for k in ("lr", "weight_decay", "dropout"):
        if k in cfg:
            arch[k] = cfg[k]
    model = advanced_models.build_neural_model(model_name, n_mels=n_mels, n_classes=n_classes, config=arch)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.to(device).eval()
    inner = model.model if hasattr(model, "model") and hasattr(model, "target_frames") else model

    def embed(X: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(np.asarray(X[i:i + 256], dtype=np.float32)).to(device)
                if inner is not model:
                    xb = torch.nn.functional.interpolate(xb, size=model.target_frames, mode="linear", align_corners=False)
                if model_name == "cnn":
                    h = inner.features(xb).mean(dim=-1)
                elif model_name == "resnet_se":
                    h = inner.blocks(inner.lift(xb)).mean(dim=-1)
                else:
                    raise ValueError(model_name)
                out.append(h.float().cpu().numpy())
        return np.concatenate(out)
    return embed


def wavlm_embedder(cfg: dict, n_classes: int, state_path: Path, audio_root: Path, device):
    import torch
    import torchaudio
    import librosa
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    enc = bundle.get_model()

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.fc = torch.nn.Linear(768, n_classes)

    model = Head()
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.to(device).eval()
    cap = int(cfg["eval_cap_seconds"] * 16000)

    def embed_paths(paths: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for p in paths:
                y, _ = librosa.load(str(audio_root / p), sr=16000, mono=True)
                y, _ = librosa.effects.trim(y, top_db=30.0)
                y = y[:cap]
                wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
                feats, _ = model.enc.extract_features(wav)
                out.append(feats[-1].mean(dim=1)[0].float().cpu().numpy())
        return np.stack(out)
    return embed_paths


# ----------------------------------------------------------------------------- probes

def speaker_probe(E: np.ndarray, spk: np.ndarray, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    speakers = sorted(set(spk.tolist()))
    y = np.asarray([speakers.index(s) for s in spk])
    n_folds = min(5, int(np.bincount(y).min()))
    if n_folds < 2:
        return {"tested": False, "reason": "fewer than 2 items for some speaker"}
    pred = np.full(len(y), -1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(E, y):
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(sc.transform(E[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(E[te]))
    recalls = [float((pred[y == c] == c).mean()) for c in range(len(speakers))]
    macro = 100.0 * float(np.mean(recalls))
    chance = 100.0 / len(speakers)
    return {"tested": True, "n_items": int(len(y)), "n_speakers": len(speakers), "cv_folds": n_folds,
            "macro_recall": macro, "chance": chance, "above_chance": macro - chance}


def nn_composition(E_train: np.ndarray, meta_train: list[dict], E_test: np.ndarray, meta_test: list[dict]) -> dict:
    a = E_train / (np.linalg.norm(E_train, axis=1, keepdims=True) + 1e-9)
    b = E_test / (np.linalg.norm(E_test, axis=1, keepdims=True) + 1e-9)
    cnt = Counter()
    for i in range(0, len(b), 512):
        sims = b[i:i + 512] @ a.T
        nn = sims.argmax(axis=1)
        for j, k in enumerate(nn):
            t, r = meta_test[i + j], meta_train[k]
            if t["speaker"] == r["speaker"] and t["sentence"] == r["sentence"] and t["label"] == r["label"]:
                cnt["sibling"] += 1
            elif t["speaker"] == r["speaker"]:
                cnt["same_speaker"] += 1
            elif t["sentence"] == r["sentence"]:
                cnt["same_sentence"] += 1
            else:
                cnt["other"] += 1
    n = sum(cnt.values())
    return {"n_test": n, **{k: cnt[k] / n for k in ("sibling", "same_speaker", "same_sentence", "other")}}


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audio-root-ravdess", type=Path)
    ap.add_argument("--audio-root-cremad", type=Path)
    ap.add_argument("--audio-root-subesco", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-ft", action="store_true")
    a = ap.parse_args(argv)
    import torch
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    io = PlanIO(a.plan, a.manifests, a.run)
    feats = FeatureStore(a.features)
    roots = {"ravdess": a.audio_root_ravdess, "cremad": a.audio_root_cremad, "subesco": a.audio_root_subesco, "subesco_980": a.audio_root_subesco}

    def units(arm, level, model, cell):
        rows = [r for r in io.plan if r["arm"] == arm and r["corpus_level"] == level and r["model"] == model and r["cell"] == cell
                and int(r["r"]) == 0 and int(r["seed_index"]) == 0]
        rows.sort(key=lambda r: int(r["fold"]))
        return [r for r in rows if (a.run / "units" / r["unit_id"] / "checkpoint.pt").exists()]

    out = {"D10": {}, "D11": {}, "definition": __doc__.strip().splitlines()[0:20]}
    specs = [("CTRL", lv, m) for lv in FAMILY for m in ("cnn", "resnet_se")]
    if not a.skip_ft:
        specs += [("FT", lv, m) for lv in FAMILY for m in ("wavlm_base_plus_ft", "wavlm_base_plus_frozen_sr")]
    for arm, level, model in specs:
        rr, gg = units(arm, level, model, "RR"), units(arm, level, model, "GG")
        if len(rr) != 5 or len(gg) != 5:
            out["D10"][f"{arm}|{level}|{model}"] = {"tested": False, "reason": f"checkpoints RR {len(rr)}/5 GG {len(gg)}/5"}
            continue
        base = rr[0]["base_corpus"]
        man = io.manifest[base]
        n_classes = io.n_classes[base]
        per_fold, comp = [], {"RR": [], "GG": []}
        for f in range(5):
            res = {"fold": f}
            emb = {}
            for cell, row in (("RR", rr[f]), ("GG", gg[f])):
                cfg = io.configs[row["config_sha256"]]
                fold = io.fold(row)
                ck = a.run / "units" / row["unit_id"] / "checkpoint.pt"
                if arm == "CTRL":
                    Xt = feats.get(base, "logmel", fold["test"])
                    embed = scratch_embedder(model, cfg, Xt.shape[1], n_classes, ck, device)
                    E_test = embed(Xt)
                    train_paths = fold["fit"] + fold["val"]
                    E_train = embed(feats.get(base, "logmel", train_paths))
                else:
                    embed_paths = wavlm_embedder(cfg, n_classes, ck, roots[base], device)
                    E_test = embed_paths(fold["test"])
                    train_paths = fold["fit"] + fold["val"]
                    E_train = embed_paths(train_paths)
                emb[cell] = (fold["test"], E_test)
                comp[cell].append(nn_composition(E_train, [man[p] for p in train_paths], E_test, [man[p] for p in fold["test"]]))
            S = sorted({man[p]["speaker"] for p in emb["GG"][0]})
            panels = {}
            for cell in ("RR", "GG"):
                paths, E = emb[cell]
                keep = [i for i, p in enumerate(paths) if man[p]["speaker"] in S]
                panels[cell] = ([paths[i] for i in keep], E[keep])
            per_spk = {s: min(sum(1 for p in panels[c][0] if man[p]["speaker"] == s) for c in ("RR", "GG")) for s in S}
            probes = {}
            for cell in ("RR", "GG"):
                paths, E = panels[cell]
                sel = []
                for s in S:
                    idx = [i for i, p in enumerate(paths) if man[p]["speaker"] == s]
                    order = _hash_order([paths[i] for i in idx], f"mechid|{level}|{model}|{cell}|{f}")
                    keepset = set(order[:per_spk[s]])
                    sel += [i for i in idx if paths[i] in keepset]
                spk = np.asarray([man[paths[i]]["speaker"] for i in sel])
                probes[cell] = speaker_probe(E[sel], spk, seed=20260903 + f)
            res.update({"speakers": S, "items_per_speaker": per_spk, "RR": probes["RR"], "GG": probes["GG"]})
            if probes["RR"].get("tested") and probes["GG"].get("tested"):
                res["RR_minus_GG_above_chance"] = probes["RR"]["above_chance"] - probes["GG"]["above_chance"]
            per_fold.append(res)
            print(f"{arm} {level} {model} fold {f}: RR {probes['RR'].get('above_chance')} GG {probes['GG'].get('above_chance')}", flush=True)
        diffs = [r["RR_minus_GG_above_chance"] for r in per_fold if "RR_minus_GG_above_chance" in r]
        out["D10"][f"{arm}|{level}|{model}"] = {"tested": bool(diffs), "folds": per_fold,
                                                 "mean_RR_minus_GG_above_chance": float(np.mean(diffs)) if diffs else None,
                                                 "mean_RR_above_chance": float(np.mean([r["RR"]["above_chance"] for r in per_fold if r["RR"].get("tested")])),
                                                 "mean_GG_above_chance": float(np.mean([r["GG"]["above_chance"] for r in per_fold if r["GG"].get("tested")]))}
        out["D11"][f"{arm}|{level}|{model}"] = {cell: {k: float(np.mean([c[k] for c in comp[cell]])) for k in ("sibling", "same_speaker", "same_sentence", "other")}
                                                 for cell in ("RR", "GG")}
    atomic_write_json(a.out, out)
    print(json.dumps({k: {kk: vv.get("mean_RR_minus_GG_above_chance") for kk, vv in v.items()} for k, v in out.items() if k == "D10"}, indent=1))


if __name__ == "__main__":
    main()
