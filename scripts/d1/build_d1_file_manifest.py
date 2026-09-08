"""P4-D1 (RAVDESS Speech-Song matched contrast): build independent file manifests.

Independently re-walks both raw sources (does NOT trust the existing P1
`ravdess_manifest.csv` blindly, per the D1 assignment's requirement to
recompute counts/keys independently) and emits one row per file with parsed
RAVDESS filename tokens:

    03-VV-EE-II-SS-RR-AA.wav
    VV = vocal channel (01 speech, 02 song)
    EE = emotion (01 neutral .. 08 surprised)
    II = intensity (01 normal, 02 strong)
    SS = statement (01, 02)
    RR = repetition (01, 02)
    AA = actor (01-24)

Restricted-speech filter (per PLAN.md revision 4 / D1 assignment section 3):
    - drop Actor 18 (has no Song counterpart)
    - keep only the six emotions common to Speech and Song
      (neutral, calm, happy, sad, angry, fearful = codes 01-06)

Sources (read-only):
    Speech: E:\\科研\\SER\\data\\Actor_*\\03-01-*.wav   (SER repo, read-only)
    Song:   E:\\claudework_data\\ICASSP2027-corpora\\ravdess-song\\Actor_*\\03-02-*.wav

Output:
    E:\\科研\\ICASSP2027\\results\\corpora\\ravdess-mode-matched\\file_manifest.csv
"""

import csv
import hashlib
import os
import sys

import soundfile as sf

SPEECH_ROOT = r"E:\科研\SER\data"
SONG_ROOT = r"E:\claudework_data\ICASSP2027-corpora\ravdess-song"
OUT_DIR = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched"
OUT_PATH = os.path.join(OUT_DIR, "file_manifest.csv")

COMMON_SIX_EMOTION_CODES = {1, 2, 3, 4, 5, 6}
EMOTION_NAMES = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}
EXCLUDED_SPEECH_ACTOR = 18


def sha256_file(path, buf_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_ravdess_filename(fname):
    """Return dict of parsed tokens, or None if filename doesn't match the
    7-field RAVDESS pattern (fail-closed: unparseable files are surfaced,
    not silently dropped)."""
    stem = fname[:-4] if fname.lower().endswith(".wav") else fname
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        modality, vocal_channel, emotion, intensity, statement, repetition, actor = (
            int(p) for p in parts
        )
    except ValueError:
        return None
    return {
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion": emotion,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
    }


def walk_channel(root, channel_label, expected_vocal_channel):
    rows = []
    unparsed = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(".wav"):
                continue
            full_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(full_path, root)
            tokens = parse_ravdess_filename(fn)
            if tokens is None:
                unparsed.append(rel_path)
                continue
            if tokens["vocal_channel"] != expected_vocal_channel:
                unparsed.append(rel_path + " (unexpected vocal_channel=%d)" % tokens["vocal_channel"])
                continue

            read_status = "ok"
            error = ""
            samplerate = frames = channels = None
            subtype = ""
            try:
                info = sf.info(full_path)
                samplerate = info.samplerate
                frames = info.frames
                channels = info.channels
                subtype = info.subtype
            except Exception as exc:  # noqa: BLE001 - want to record any failure
                read_status = "error"
                error = str(exc)

            rows.append({
                "channel": channel_label,
                "relative_path": rel_path.replace("\\", "/"),
                "file_size_bytes": os.path.getsize(full_path),
                "sha256": sha256_file(full_path),
                "actor": tokens["actor"],
                "emotion_code": tokens["emotion"],
                "emotion_name": EMOTION_NAMES.get(tokens["emotion"], "unknown"),
                "intensity": tokens["intensity"],
                "statement": tokens["statement"],
                "repetition": tokens["repetition"],
                "modality": tokens["modality"],
                "vocal_channel": tokens["vocal_channel"],
                "wav_samplerate": samplerate,
                "wav_frames": frames,
                "wav_channels": channels,
                "wav_subtype": subtype,
                "read_status": read_status,
                "error": error,
            })
    return rows, unparsed


def main():
    song_rows, song_unparsed = walk_channel(SONG_ROOT, "song", expected_vocal_channel=2)

    speech_rows_all, speech_unparsed = walk_channel(SPEECH_ROOT, "speech", expected_vocal_channel=1)
    # restricted-speech filter: common six emotions, drop Actor 18
    speech_rows = [
        r for r in speech_rows_all
        if r["emotion_code"] in COMMON_SIX_EMOTION_CODES and r["actor"] != EXCLUDED_SPEECH_ACTOR
    ]
    speech_dropped_actor18 = [r for r in speech_rows_all if r["actor"] == EXCLUDED_SPEECH_ACTOR]
    speech_dropped_uncommon_emotion = [
        r for r in speech_rows_all
        if r["actor"] != EXCLUDED_SPEECH_ACTOR and r["emotion_code"] not in COMMON_SIX_EMOTION_CODES
    ]

    print("song: %d files parsed, %d unparsed" % (len(song_rows), len(song_unparsed)))
    if song_unparsed:
        print("  song unparsed sample:", song_unparsed[:10])
    print("speech (raw, all 24 actors x 8 emotions): %d files parsed, %d unparsed"
          % (len(speech_rows_all), len(speech_unparsed)))
    if speech_unparsed:
        print("  speech unparsed sample:", speech_unparsed[:10])
    print("speech restricted (common six emotions, Actor 18 dropped): %d files" % len(speech_rows))
    print("  dropped for Actor 18: %d" % len(speech_dropped_actor18))
    print("  dropped for disgust/surprised (uncommon emotions): %d" % len(speech_dropped_uncommon_emotion))

    song_actor_count = len(set(r["actor"] for r in song_rows))
    speech_actor_count = len(set(r["actor"] for r in speech_rows))
    print("song distinct actors: %d" % song_actor_count)
    print("speech (restricted) distinct actors: %d" % speech_actor_count)

    song_read_errors = [r for r in song_rows if r["read_status"] != "ok"]
    speech_read_errors = [r for r in speech_rows if r["read_status"] != "ok"]
    print("song read errors: %d" % len(song_read_errors))
    print("speech (restricted) read errors: %d" % len(speech_read_errors))

    os.makedirs(OUT_DIR, exist_ok=True)
    fieldnames = [
        "channel", "relative_path", "file_size_bytes", "sha256", "actor",
        "emotion_code", "emotion_name", "intensity", "statement", "repetition",
        "modality", "vocal_channel", "wav_samplerate", "wav_frames",
        "wav_channels", "wav_subtype", "read_status", "error",
    ]
    all_rows = song_rows + speech_rows
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print("wrote %d rows to %s" % (len(all_rows), OUT_PATH))

    # Hard-fail signal for the calling shell: nonzero exit if counts are off
    # spec (23 actors x 44 files = 1012 per channel) or any unparsed/error file
    # exists, so this step cannot silently pass a bad manifest.
    ok = (
        len(song_rows) == 1012
        and len(speech_rows) == 1012
        and song_actor_count == 23
        and speech_actor_count == 23
        and not song_unparsed
        and not speech_unparsed
        and not song_read_errors
        and not speech_read_errors
    )
    print("GATE_2_3_FILE_MANIFEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
