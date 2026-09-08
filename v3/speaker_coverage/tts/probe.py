"""Fixed-attempt VoiceDesign engineering probe; never trains or scores SER.

The freeze/blind commands do not import torch or qwen_tts. Generated anchors
are comparison items, not conditioning audio. Repeating a description is not
an assurance of speaker identity. See README.md for the deliberately narrow aim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
import time

import numpy as np


SCHEMA = "ser-tts-voicedesign-probe-1"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
PACKAGE = "qwen-tts==0.1.1"
MASTER_SEED = 20260906
TEXT_SOURCE = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/README.md"
API_SOURCE = "https://github.com/QwenLM/Qwen3-TTS#voice-design"
TEXTS = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}
VOICES = (
    {"voice_id": "F1", "sex_description": "female", "description":
     "An entirely fictional adult woman in her thirties, speaking English with a General American accent. "
     "A low-mid register, rounded resonant timbre, a lightly grainy texture, and clear consonants."},
    {"voice_id": "F2", "sex_description": "female", "description":
     "An entirely fictional adult woman in her forties, speaking English with a General American accent. "
     "A mid-high register, light clear timbre, a smooth texture, and crisp consonants."},
    {"voice_id": "M1", "sex_description": "male", "description":
     "An entirely fictional adult man in his thirties, speaking English with a General American accent. "
     "A low register, broad chest resonance, a lightly rough texture, and clear consonants."},
    {"voice_id": "M2", "sex_description": "male", "description":
     "An entirely fictional adult man in his forties, speaking English with a General American accent. "
     "A mid register, light forward resonance, a smooth texture, and crisp consonants."},
)
EMOTIONS = {
    "ANG": "Deliver the sentence with clearly audible anger, irritation, and tense emphasis.",
    "DIS": "Deliver the sentence with clearly audible disgust and aversion.",
    "FEA": "Deliver the sentence with clearly audible fear, apprehension, and vocal tension.",
    "HAP": "Deliver the sentence with clearly audible happiness and lively, cheerful intonation.",
    "NEU": "Deliver the sentence neutrally, in a matter-of-fact tone without a strong emotion.",
    "SAD": "Deliver the sentence with clearly audible sadness and subdued, sorrowful intonation.",
}
GENERATION = {"max_new_tokens": 512, "do_sample": True, "top_p": 0.9,
              "top_k": 50, "temperature": 0.8, "repetition_penalty": 1.05}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable_json(path, value):
    """Atomic idempotent creation; never replace a different existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == content, f"existing artifact differs: {path}")
    else:
        fd, temporary = tempfile.mkstemp(prefix=".pending-json-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Atomic no-clobber publication, on both NTFS and Linux.
                os.link(temporary, path)
            except FileExistsError:
                require(path.read_text(encoding="utf-8") == content, f"existing artifact differs: {path}")
        finally:
            Path(temporary).unlink(missing_ok=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_relative(name):
    require(isinstance(name, str) and bool(name), "nonempty relative path required")
    p = PurePosixPath(name)
    require(not p.is_absolute() and str(p) == name and ".." not in p.parts
            and "\\" not in name and ":" not in name, "unsafe relative path")
    return p


def verify_model(model_dir, pins_path):
    """Byte-check the whole already-downloaded snapshot; no network fallback."""
    directory = Path(model_dir).resolve(strict=True)
    pins = read_json(pins_path)
    require(pins.get("model_id") == MODEL_ID and pins.get("package") == PACKAGE,
            "only the pinned VoiceDesign model/package is supported")
    require(re.fullmatch(r"[0-9a-f]{40}", pins.get("revision", "")), "immutable model revision required")
    expected = pins.get("files")
    require(isinstance(expected, dict) and bool(expected), "model file pins required")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and ".cache" not in p.relative_to(directory).parts}
    require(actual == set(expected), "model snapshot file inventory differs from pins (excluding .cache)")
    for name, info in sorted(expected.items()):
        rel = safe_relative(name)
        require(".cache" not in rel.parts and isinstance(info, dict), "invalid model file pin")
        path = directory.joinpath(*rel.parts)
        require(path.stat().st_size == info.get("bytes") and file_sha(path) == info.get("sha256"),
                f"model bytes differ from pin: {name}")
    return {"model_id": MODEL_ID, "revision": pins["revision"], "package": PACKAGE,
            "pins_file_sha256": file_sha(pins_path), "files": expected,
            "snapshot_sha256": digest(expected)}


def sentence_code(path):
    safe_relative(path)
    parts = PurePosixPath(path).name.split("_")
    require(len(parts) == 4 and parts[0].isdigit() and parts[1] in TEXTS
            and parts[2] in EMOTIONS and parts[3].endswith(".wav"), "unexpected CREMA-D audio name")
    return parts[1]


def prompt_context(plan):
    require(plan.get("schema") == "ser-speaker-coverage-1", "coverage plan schema mismatch")
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "coverage plan hash mismatch")
    selected = [u for u in plan["units"] if (u.get("model"), u.get("policy"), u.get("fold"),
                u.get("rotation"), u.get("draw")) == ("cnn", "U", 0, 0, 0)]
    require(len(selected) == 1, "unique U/CNN/fold0/rotation0/draw0 source unit required")
    unit = selected[0]
    require(unit.get("unit_id") == digest({k: v for k, v in unit.items() if k != "unit_id"}),
            "source unit hash mismatch")
    train = sorted({sentence_code(p) for p in unit["fit"]})
    query = sorted({sentence_code(p) for p in unit["test"]})
    stop = sorted({sentence_code(p) for p in unit["val"]})
    require(len(train) == 8 and len(query) == 2 and len(stop) == 6
            and set(query).isdisjoint(train) and set(stop) <= set(train), "source prompt roles invalid")
    return {"fold": 0, "rotation": 0, "source_unit_id": unit["unit_id"],
            "allowed_train_prompts": train, "query_prompts": query, "stop_prompts": stop,
            "probe_prompts": train[:4], "selection_rule": "lexicographically_first_four_of_allowed_eight"}


def build_manifest(plan, plan_file_sha256, model_identity, source_sha256):
    context = prompt_context(plan)
    attempts = []
    for voice in VOICES:
        # One extra NEU rendition of a training sentence is an identity anchor.
        # It is not supplied to the model when generating the other 24 items.
        cells = [("probe", emotion, prompt) for emotion in EMOTIONS for prompt in context["probe_prompts"]]
        cells.append(("anchor", "NEU", context["probe_prompts"][0]))
        for role, emotion, prompt in cells:
            attempt_id = f"a{len(attempts) + 1:03d}"
            seed = int(digest([MASTER_SEED, "tts-probe", voice["voice_id"], role, emotion, prompt])[:8], 16)
            row = {"attempt_id": attempt_id, "voice_id": voice["voice_id"], "role": role,
                   "emotion": emotion, "prompt": prompt, "text": TEXTS[prompt], "seed": seed,
                   "language": "English", "voice_description_sha256": digest(voice["description"]),
                   "instruct": voice["description"] + " " + EMOTIONS[emotion]
                   + " Speak only the provided words. Use the described adult voice.",
                   "generation": dict(GENERATION), "conditioning_audio": None}
            row["attempt_sha256"] = digest(row)
            attempts.append(row)
    require(len(attempts) == 100 and len({a["seed"] for a in attempts}) == 100, "100 unique seeded attempts required")
    result = {"schema": SCHEMA, "scope": "technical_description_identity_probe_only",
              "source_sha256": source_sha256, "plan_sha256": plan["plan_sha256"],
              "plan_file_sha256": plan_file_sha256, "context": context,
              "synthetic_fixture": bool(plan.get("design", {}).get("fixture", False)),
              "model": model_identity, "runtime": {"dtype": "bfloat16", "attention": "sdpa",
                  "batch_size": 1, "device": "cuda:0", "package": PACKAGE, "offline": True},
              "references": {"audio_conditioning": "none", "anchors": "generated_comparison_only",
                  "text_source": TEXT_SOURCE, "text_mapping_sha256": digest(TEXTS),
                  "text_source_revision": "not_pinned_remote_main;literal_mapping_frozen_by_hash",
                  "api_source": API_SOURCE},
              "voices": list(VOICES), "attempt_count": 100,
              "attempt_policy": "one_call_each;terminal_failures_and_interruptions;no_quality_replacement",
              "attempts": attempts}
    result["manifest_sha256"] = digest(result)
    return result


def load_manifest(path, check_source=True):
    value = read_json(path)
    require(value.get("schema") == SCHEMA and value.get("manifest_sha256") ==
            digest({k: v for k, v in value.items() if k != "manifest_sha256"}), "probe manifest hash mismatch")
    require(value.get("attempt_count") == len(value.get("attempts", [])) == 100, "probe count differs")
    require([a["attempt_id"] for a in value["attempts"]] == [f"a{i:03d}" for i in range(1, 101)],
            "attempt identity/order differs")
    for attempt in value["attempts"]:
        require(attempt["attempt_sha256"] == digest({k: v for k, v in attempt.items() if k != "attempt_sha256"}),
                "attempt content hash mismatch")
        require(attempt["generation"] == GENERATION and attempt["conditioning_audio"] is None,
                "generation settings/conditioning differ")
    if check_source:
        require(value["source_sha256"] == file_sha(__file__), "probe source changed after freeze; use the frozen source")
    return value


def config_for(manifest, attempt):
    return {"manifest_sha256": manifest["manifest_sha256"], "model": {
        k: manifest["model"][k] for k in ("model_id", "revision", "snapshot_sha256", "pins_file_sha256")},
        "source_sha256": manifest["source_sha256"], "runtime": manifest["runtime"], "attempt": attempt}


def freeze(plan_path, model_dir, pins_path, out):
    identity = verify_model(model_dir, pins_path)
    manifest = build_manifest(read_json(plan_path), file_sha(plan_path), identity, file_sha(__file__))
    out = Path(out)
    # Every config is materialized before generation is possible.
    for attempt in manifest["attempts"]:
        immutable_json(out / "attempts" / attempt["attempt_id"] / "config.json", config_for(manifest, attempt))
    immutable_json(out / "manifest.json", manifest)
    return manifest


def waveform_metrics(waveform, sample_rate):
    values = np.asarray(waveform, dtype=np.float32)
    finite = bool(np.isfinite(values).all())
    mono = values.ndim == 1
    count = int(values.size)
    valid_rate = isinstance(sample_rate, (int, np.integer)) and not isinstance(sample_rate, bool) and sample_rate > 0
    valid = mono and count > 0 and finite and valid_rate
    # JSON never contains NaN; failed numerical metrics are explicitly absent.
    return values, {"finite": finite, "mono": mono, "samples": count,
        "sample_rate": int(sample_rate) if valid_rate else None,
        "duration_seconds": count / int(sample_rate) if mono and valid_rate else None,
        "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) if finite and count else None,
        "peak_abs": float(np.max(np.abs(values))) if finite and count else None,
        "clip_fraction_abs_ge_0_999": float(np.mean(np.abs(values) >= .999)) if finite and count else None,
        "technically_writable": bool(valid)}


def existing_receipt(directory, manifest, attempt):
    path = directory / "receipt.json"
    if not path.exists():
        return None
    value = read_json(path)
    require(value.get("manifest_sha256") == manifest["manifest_sha256"]
            and value.get("attempt_sha256") == attempt["attempt_sha256"]
            and value.get("config_sha256") == file_sha(directory / "config.json")
            and value.get("status") in ("success", "failed", "interrupted"), "existing attempt receipt differs")
    if value["status"] == "success":
        require(value.get("wav_file") == "audio.wav" and file_sha(directory / "audio.wav") == value.get("wav_sha256"),
                "completed WAV changed or is missing; never silently regenerate")
    return value


@contextmanager
def generation_lock(out):
    """OS advisory lock releases on process exit; no unsafe stale-lock deletion."""
    path = Path(out) / ".generation.lock"
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class QwenBackend:
    def __init__(self, model_dir):
        # Set before importing transformers/qwen. A missing local file fails.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        require(importlib.metadata.version("qwen-tts") == "0.1.1", "qwen-tts version differs from frozen requirement")
        import torch
        from qwen_tts import Qwen3TTSModel
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "one bf16-capable CUDA GPU required")
        self.torch = torch
        self.model = Qwen3TTSModel.from_pretrained(str(Path(model_dir).resolve(strict=True)),
            device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        self.environment = {"backend": "Qwen3TTSModel.generate_voice_design", "qwen-tts": "0.1.1",
            "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "pid": os.getpid(),
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate")}

    def start_attempt(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats(0)

    def generate(self, attempt):
        with self.torch.inference_mode():
            wavs, sr = self.model.generate_voice_design(text=attempt["text"], language="English",
                instruct=attempt["instruct"], **attempt["generation"])
        self.torch.cuda.synchronize()
        require(len(wavs) == 1, "batch size must remain exactly one")
        wave = wavs[0]
        if hasattr(wave, "detach"):
            wave = wave.detach().float().cpu().numpy()
        return wave, sr

    def memory(self):
        return {"peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved(0))}


def run_attempts(manifest, out, backend):
    """One call per not-yet-started item. Tests use an explicitly synthetic backend."""
    import soundfile as sf
    out = Path(out)
    receipts = []
    for attempt in manifest["attempts"]:
        directory = out / "attempts" / attempt["attempt_id"]
        immutable_json(directory / "config.json", config_for(manifest, attempt))
        receipt = existing_receipt(directory, manifest, attempt)
        if receipt is not None:
            receipts.append(receipt)
            continue
        receipt = {"attempt_id": attempt["attempt_id"], "attempt_sha256": attempt["attempt_sha256"],
            "manifest_sha256": manifest["manifest_sha256"], "config_sha256": file_sha(directory / "config.json"),
            "seed": attempt["seed"], "environment": backend.environment,
            "wav_file": None, "wav_sha256": None, "metrics": None, "vram": None,
            "wall_seconds": None, "error": None}
        started = directory / "started.json"
        if started.exists():
            prior = read_json(started)
            require(prior.get("attempt_sha256") == attempt["attempt_sha256"]
                    and prior.get("manifest_sha256") == manifest["manifest_sha256"], "started receipt differs")
            receipt.update(status="interrupted", started_utc=prior["started_utc"], finished_utc=utc_now(),
                           error="Previously started without a terminal receipt; no retry permitted.")
            receipt["orphan_wav_sha256"] = file_sha(directory / "audio.wav") if (directory / "audio.wav").exists() else None
        else:
            receipt["started_utc"] = utc_now()
            immutable_json(started, {"started_utc": receipt["started_utc"],
                "manifest_sha256": manifest["manifest_sha256"], "attempt_sha256": attempt["attempt_sha256"]})
            begun = time.perf_counter()
            try:
                backend.start_attempt(attempt["seed"])
                raw, sr = backend.generate(attempt)
                samples, metrics = waveform_metrics(raw, sr)
                receipt["metrics"] = metrics
                require(metrics["technically_writable"], "empty, non-mono, non-finite, or invalid sample-rate output")
                # FLOAT preserves raw values: no normalization or clipping repair.
                sf.write(directory / "audio.wav", samples, int(sr), subtype="FLOAT")
                receipt.update(status="success", wav_file="audio.wav", wav_sha256=file_sha(directory / "audio.wav"))
            except Exception as error:
                receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            finally:
                # Interrupt/kill leaves started.json terminalized on the next run.
                receipt["wall_seconds"] = time.perf_counter() - begun
                receipt["finished_utc"] = utc_now()
                try:
                    receipt["vram"] = backend.memory()
                except Exception as error:
                    receipt["memory_error"] = f"{type(error).__name__}: {error}"
        immutable_json(directory / "receipt.json", receipt)
        receipts.append(receipt)
        print(canonical({k: receipt[k] for k in ("attempt_id", "status", "wall_seconds", "metrics")}), flush=True)
    return receipts


def terminal_receipts(manifest, out):
    values = []
    for attempt in manifest["attempts"]:
        directory = Path(out) / "attempts" / attempt["attempt_id"]
        immutable_json(directory / "config.json", config_for(manifest, attempt))
        receipt = existing_receipt(directory, manifest, attempt)
        require(receipt is not None, f"attempt not terminal: {attempt['attempt_id']}")
        values.append(receipt)
    return values


def generate(manifest_path, model_dir, pins_path):
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    require(not manifest["synthetic_fixture"], "synthetic fixture cannot be sent to a real model")
    out = manifest_path.parent
    with generation_lock(out):
        require(verify_model(model_dir, pins_path) == manifest["model"], "model identity differs from frozen manifest")
        complete = all((out / "attempts" / a["attempt_id"] / "receipt.json").exists() for a in manifest["attempts"])
        if complete:
            receipts = terminal_receipts(manifest, out)
        else:
            backend = QwenBackend(model_dir)
            receipts = run_attempts(manifest, out, backend)
        summary = {"manifest_sha256": manifest["manifest_sha256"], "attempts": len(receipts),
                   "status_counts": dict(Counter(r["status"] for r in receipts)),
                   "receipt_sha256": {r["attempt_id"]: file_sha(out / "attempts" / r["attempt_id"] / "receipt.json")
                                      for r in receipts},
                   "scope": "engineering_only_no_human_ratings_no_SER_no_identity_success_claim"}
        immutable_json(out / "generation_summary.json", summary)
    return summary


def write_csv(path, columns, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def blind(manifest_path):
    """Export anonymized files and EMPTY rating forms, never invented ratings."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    out = manifest_path.parent
    receipts = terminal_receipts(manifest, out)
    success = {r["attempt_id"] for r in receipts if r["status"] == "success"}
    destination = out / "blind"
    require(not destination.exists(), "blind output already exists; preserve any human-entered ratings")
    destination.mkdir()
    private = []
    for rater in range(1, 4):
        folder = destination / f"rater_{rater}"
        audio = folder / "audio"
        audio.mkdir(parents=True)
        items = sorted((a for a in manifest["attempts"] if a["attempt_id"] in success),
                       key=lambda a: digest([MASTER_SEED, "blind-order", rater, a["attempt_id"]]))
        ids = {a["attempt_id"]: "clip_" + digest([manifest["manifest_sha256"], "blind-id", rater,
                                               a["attempt_id"]])[:16] for a in items}
        anchors = [a for a in items if a["role"] == "anchor"]
        references = [ids[a["attempt_id"]] for a in anchors]
        quality, identity = [], []
        for item in items:
            anon = ids[item["attempt_id"]]
            source = out / "attempts" / item["attempt_id"] / "audio.wav"
            shutil.copyfile(source, audio / (anon + ".wav"))
            quality.append({"clip_id": anon, "audio_file": f"audio/{anon}.wav",
                "perceived_emotion_ANG_DIS_FEA_HAP_NEU_SAD_OTHER_UNCLEAR": "", "naturalness_1_to_5": "",
                "intelligibility_1_to_5": "", "comments": ""})
            if item["role"] == "probe":
                identity.append({"clip_id": anon, "audio_file": f"audio/{anon}.wav",
                    "reference_clip_ids": ";".join(references), "best_matching_reference_id_or_NONE_UNCLEAR": "",
                    "identity_confidence_1_to_5": "", "comments": ""})
            private.append({"rater": rater, "clip_id": anon, "attempt_id": item["attempt_id"],
                "voice_id": item["voice_id"], "intended_emotion": item["emotion"], "role": item["role"],
                "text": item["text"], "wav_sha256": file_sha(source),
                "same_description_anchor_id": next((ids[a["attempt_id"]] for a in anchors
                                                      if a["voice_id"] == item["voice_id"]), None)})
        write_csv(folder / "quality_ratings.csv", ["clip_id", "audio_file",
            "perceived_emotion_ANG_DIS_FEA_HAP_NEU_SAD_OTHER_UNCLEAR", "naturalness_1_to_5",
            "intelligibility_1_to_5", "comments"], quality)
        write_csv(folder / "identity_ratings.csv", ["clip_id", "audio_file", "reference_clip_ids",
            "best_matching_reference_id_or_NONE_UNCLEAR", "identity_confidence_1_to_5", "comments"], identity)
        (folder / "INSTRUCTIONS.txt").write_text(
            "Pending independent human evaluation. Work alone, without target labels or other raters' answers.\n"
            "First rate all clips in quality_ratings.csv in the given order. ANG=anger, DIS=disgust, FEA=fear, "
            "HAP=happiness, NEU=neutral, SAD=sadness. OTHER/UNCLEAR are valid responses.\n"
            "Naturalness: 1 very artificial to 5 fully natural. Intelligibility: 1 unintelligible to 5 all words clear.\n"
            "Then compare each identity item with the listed reference clips, whose filenames are in audio/. "
            "Choose the closest perceived speaker identity, or NONE/UNCLEAR if there is no clear match. "
            "Identity confidence: 1 low to 5 high. Do not assume any reference is the same speaker.\n"
            "Do not infer intended emotion from filenames: all filenames are anonymized. "
            "Do not discuss ratings until all three raters have completed their forms.\n", encoding="utf-8")
    immutable_json(out / "blind_private_key.json", {"manifest_sha256": manifest["manifest_sha256"], "items": private})
    summary = {"manifest_sha256": manifest["manifest_sha256"], "status": "awaiting_three_independent_human_raters",
        "raters_completed": 0, "success_clips_per_rater": len(success),
        "omitted_attempts": [r["attempt_id"] for r in receipts if r["status"] != "success"],
        "missing_anchors": [a["attempt_id"] for a in manifest["attempts"] if a["role"] == "anchor" and a["attempt_id"] not in success],
        "share_only": "one blind/rater_N directory per person; never share manifest/config/private key",
        "asv_status": "not_implemented_not_run;requires_separately_pinned_embedding_analysis"}
    immutable_json(out / "blind_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze", help="CPU-only fixed manifest and source/model hash freeze")
    freeze_parser.add_argument("--plan", type=Path, required=True)
    freeze_parser.add_argument("--model-dir", type=Path, required=True)
    freeze_parser.add_argument("--model-pins", type=Path, required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    gen_parser = commands.add_parser("generate", help="single-GPU generation, only after freeze")
    gen_parser.add_argument("--manifest", type=Path, required=True)
    gen_parser.add_argument("--model-dir", type=Path, required=True)
    gen_parser.add_argument("--model-pins", type=Path, required=True)
    blind_parser = commands.add_parser("blind", help="empty rating forms for three independent human raters")
    blind_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.plan, args.model_dir, args.model_pins, args.out)
        print(canonical({"manifest_sha256": result["manifest_sha256"], "attempt_count": result["attempt_count"],
                         "probe_prompts": result["context"]["probe_prompts"]}))
    elif args.command == "generate":
        print(canonical(generate(args.manifest, args.model_dir, args.model_pins)))
    else:
        print(canonical(blind(args.manifest)))


if __name__ == "__main__":
    main()
