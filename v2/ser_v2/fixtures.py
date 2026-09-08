"""Synthetic run generator. Produces a complete, contract-conformant run (predictions.csv,
unit.json, history.json, DONE per unit) from a latent 'world' with known effects, so that the
scorer and the independent verifier can be compared and the mutation battery has a target.

    python -m ser_v2.fixtures --plan <plan dir> --manifests <dir> --out <run dir> [--arms CTRL,MECH2X2] [--levels ravdess,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import corpora
from .common import atomic_write_json, atomic_write_text, read_csv, read_json, sha256_text, stable_u32

# world parameters: additive shifts on the per-utterance probability of a correct prediction
DEFAULT_WORLD = {
    "base_correct": {"ravdess": 0.45, "cremad": 0.55, "subesco": 0.50},
    "model_shift": {"cnn": 0.0, "resnet_se": 0.05, "transformer": -0.02, "hubert_base": 0.03, "wavlm_base_plus": 0.02,
                    "wav2vec2_base": 0.01, "wavlm_base_plus_ft": 0.10, "wavlm_base_plus_frozen_sr": 0.04,
                    "ridge_a1_hubert_base": 0.02, "ridge_a1_wavlm_base_plus": 0.01, "ridge_a1_wav2vec2_base": 0.0},
    # leakage boosts by cell (pipeline-level): outer overlap and inner overlap
    "cell_shift": {"RR": 0.16, "RG": 0.10, "GR": 0.03, "GG": 0.0, "TG": 0.02, "LOSO": 0.01, "LOSOSUB": 0.0,
                   "P1RR": 0.16, "P1GG": 0.0, "RO": 0.14, "GO": 0.0, "G5": 0.0,
                   "RR_hpo": 0.18, "GR_hpo": 0.03, "GG_hpo": 0.0,
                   "none": 0.0, "spk": 0.05, "prm": 0.03, "both": 0.08, "both_sib": 0.14, "spk_half_h1": 0.05, "spk_half_h2": 0.05},
    "corpus_leak_scale": {"ravdess": 1.0, "cremad": 0.2, "subesco": 0.5},
    "speaker_sd": 0.08, "replicate_sd": 0.02, "utterance_noise": 0.0,
    "ft_reexposure": 0.04,           # extra RR boost for the fine-tuned model
    "hpo_val_optimism_random_inner": 0.06, "hpo_config_effect_sd": 0.02,
}


def _rng(*parts) -> np.random.RandomState:
    return np.random.RandomState(stable_u32("fixture|" + "|".join(str(p) for p in parts)))


def load_world(path: Path | None) -> dict:
    w = json.loads(json.dumps(DEFAULT_WORLD))
    if path:
        w.update(read_json(path))
    return w


def null_world() -> dict:
    w = json.loads(json.dumps(DEFAULT_WORLD))
    w["cell_shift"] = {k: 0.0 for k in w["cell_shift"]}
    w["ft_reexposure"] = 0.0
    w["hpo_val_optimism_random_inner"] = 0.0
    return w


def generate(plan_dir: Path, manifests_dir: Path, out: Path, world: dict, arms: set[str] | None = None,
             levels: set[str] | None = None, run_id: str = "synthetic") -> dict:
    plan = read_csv(plan_dir / "run_plan.csv")
    split_index = read_json(plan_dir / "split_index.json")
    configs = read_json(plan_dir / "unit_configs.json")
    hygiene_rows: dict[str, dict[str, dict]] = {}
    for base in ("ravdess", "cremad", "subesco", "subesco_980"):
        p = manifests_dir / f"{base}_manifest.csv"
        if p.exists():
            rows = corpora.load_manifest(p)
            rows, _ = corpora.apply_hygiene("subesco" if base == "subesco_980" else base, rows)
        else:
            rows = corpora.synthetic_manifest(base, 20, 10, 5)
        hygiene_rows[base] = {r["relative_path"]: r for r in rows}
    n_classes = {base: len(corpora.CORPORA["subesco" if base == "subesco_980" else base].labels) for base in hygiene_rows}
    units_dir = out / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    split_cache: dict[str, dict] = {}
    n_done = 0
    for u in plan:
        if arms and u["arm"] not in arms:
            continue
        if levels and u["corpus_level"].split("_d")[0] not in levels and u["corpus_level"] not in levels:
            continue
        key = next(k for k, v in split_index.items() if v["sha256"] == u["split_sha256"])
        if key not in split_cache:
            split_cache[key] = read_json(plan_dir / "splits" / f"{key}.json")
        table = split_cache[key]
        fold = next(f for f in table["folds"] if f["fold"] == int(u["fold"]))
        base = u["base_corpus"]
        lookup = hygiene_rows[base]
        K = n_classes[base]
        corpus_key = "subesco" if base.startswith("subesco") else base
        w = world
        rng = _rng(u["unit_id"])
        level_root = u["corpus_level"].split("_d")[0]
        leak = w["cell_shift"].get(u["cell"], 0.0) * w["corpus_leak_scale"][corpus_key]
        if u["model"] == "wavlm_base_plus_ft" and u["cell"] == "RR":
            leak += w["ft_reexposure"]
        cfg = configs[u["config_sha256"]]
        cfg_effect = 0.0
        base_p = w["base_correct"][corpus_key] + w["model_shift"].get(u["model"], 0.0) + leak
        rep_shift = _rng("rep", level_root, u["model"], u["cell"], u["r"], u["seed_index"]).normal(0, w["replicate_sd"])
        rows_out = []
        for path in fold["test"]:
            m = lookup[path]
            spk_shift = _rng("spk", corpus_key, m["speaker"]).normal(0, w["speaker_sd"])
            p_correct = float(np.clip(base_p + spk_shift + rep_shift, 0.02, 0.98))
            y = int(m["label_index"])
            if rng.rand() < p_correct:
                pred = y
            else:
                pred = int(rng.choice([c for c in range(K) if c != y]))
            logits = rng.normal(0, 0.3, size=K)
            logits[pred] = logits.max() + 1.0 + rng.rand()
            rows_out.append((int(m["sample_index"]), path, m["speaker"], y, pred, logits))
        rows_out.sort(key=lambda t: t[0])
        lines = ["sample_index,relative_path,speaker,y_true,y_pred," + ",".join(f"logit_{i}" for i in range(K))]
        for si, path, spk, y, pred, logits in rows_out:
            lines.append(f"{si},{path},{spk},{y},{pred}," + ",".join(repr(float(v)) for v in logits))
        text = "\n".join(lines) + "\n"
        sha = sha256_text(text)
        udir = units_dir / u["unit_id"]
        udir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(udir / "predictions.csv", text)
        # validation summary (inner leakage optimism for random-inner cells; config effects for HPO)
        val_opt = 0.0
        if u["cell"] in ("RR", "GR", "RR_hpo", "GR_hpo", "P1RR"):
            val_opt = w["hpo_val_optimism_random_inner"] if u["arm"] == "HPO" else 0.03
        if u["arm"] == "HPO":
            cfg_effect = _rng("hpo", level_root, u["fold"], cfg.get("hpo_config_index")).normal(0, w["hpo_config_effect_sd"])
        val_uar = 100.0 * float(np.clip(base_p + val_opt + cfg_effect + rep_shift, 0.02, 0.98))
        unit_json = {"unit_id": u["unit_id"], "arm": u["arm"], "corpus_level": u["corpus_level"], "model": u["model"],
                     "cell": u["cell"], "fold": int(u["fold"]), "r": int(u["r"]), "seed_index": int(u["seed_index"]),
                     "train_seed": int(u["train_seed"]), "config_sha256": u["config_sha256"], "split_sha256": u["split_sha256"],
                     "predictions_sha256": sha, "n_fit": int(u["n_fit"]), "n_val": int(u["n_val"]), "n_test": int(u["n_test"]),
                     "best_epoch": int(rng.randint(5, 40)), "epochs_run": int(rng.randint(20, 60)),
                     "val_uar_best": val_uar, "val_loss_best": float(rng.uniform(0.8, 1.6)),
                     "gpu_seconds": float(u["est_gpu_sec"]), "started_at": "synthetic", "finished_at": "synthetic",
                     "resumed": False, "engine_version": "fixture-1", "status": "done", "config": cfg}
        atomic_write_json(udir / "unit.json", unit_json)
        atomic_write_json(udir / "history.json", [{"epoch": e, "train_loss": 1.5 - 0.01 * e, "val_loss": 1.4 - 0.008 * e, "val_uar": val_uar - 5 + 0.1 * e} for e in range(1, 6)])
        atomic_write_text(udir / "DONE", sha + "\n")
        n_done += 1
    atomic_write_json(out / "run.json", {"run_id": run_id, "plan_dir": str(plan_dir), "world": world, "n_units_done": n_done, "synthetic": True})
    return {"n_units_done": n_done}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--null-world", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--levels", default="")
    a = ap.parse_args(argv)
    world = null_world() if a.null_world else load_world(a.world)
    info = generate(a.plan, a.manifests, a.out, world, set(a.arms.split(",")) if a.arms else None,
                    set(a.levels.split(",")) if a.levels else None)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
