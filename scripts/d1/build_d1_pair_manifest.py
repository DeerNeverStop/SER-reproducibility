"""P4-D1: build pair_manifest.csv from file_manifest.csv and verify the
1,012-key bidirectional match required by the D1 assignment (section 3,
"1,012 个 key、双向差集为 0").

Pair key = (actor, emotion_code, intensity, statement, repetition).
Fails closed (nonzero exit, no pair_manifest.csv written) if:
  - either side has a duplicate key (many-to-one),
  - the two-way key set difference is non-empty,
  - the resulting join is not exactly 1,012 rows.
"""

import csv
import os
import sys
from collections import Counter

IN_PATH = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched\file_manifest.csv"
OUT_PATH = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched\pair_manifest.csv"


def load_rows():
    with open(IN_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_of(row):
    return (
        int(row["actor"]), int(row["emotion_code"]), int(row["intensity"]),
        int(row["statement"]), int(row["repetition"]),
    )


def main():
    rows = load_rows()
    song = [r for r in rows if r["channel"] == "song"]
    speech = [r for r in rows if r["channel"] == "speech"]

    song_keys = [key_of(r) for r in song]
    speech_keys = [key_of(r) for r in speech]

    song_dupe = [k for k, c in Counter(song_keys).items() if c > 1]
    speech_dupe = [k for k, c in Counter(speech_keys).items() if c > 1]
    print("song duplicate keys:", len(song_dupe))
    print("speech duplicate keys:", len(speech_dupe))

    song_set = set(song_keys)
    speech_set = set(speech_keys)
    only_in_song = song_set - speech_set
    only_in_speech = speech_set - song_set
    print("keys only in song (not in speech):", len(only_in_song))
    print("keys only in speech (not in song):", len(only_in_speech))
    if only_in_song:
        print("  sample:", list(only_in_song)[:10])
    if only_in_speech:
        print("  sample:", list(only_in_speech)[:10])

    fail_closed = bool(song_dupe or speech_dupe or only_in_song or only_in_speech)
    if fail_closed:
        print("GATE_3_PAIR_KEY: FAIL (fail-closed, pair_manifest.csv NOT written)")
        return 1

    song_by_key = {key_of(r): r for r in song}
    speech_by_key = {key_of(r): r for r in speech}

    fieldnames = [
        "actor", "emotion_code", "emotion_name", "intensity", "statement", "repetition",
        "song_relative_path", "song_sha256", "song_file_size_bytes",
        "speech_relative_path", "speech_sha256", "speech_file_size_bytes",
    ]
    out_rows = []
    for k in sorted(song_set):
        s = song_by_key[k]
        p = speech_by_key[k]
        out_rows.append({
            "actor": k[0], "emotion_code": k[1], "emotion_name": s["emotion_name"],
            "intensity": k[2], "statement": k[3], "repetition": k[4],
            "song_relative_path": s["relative_path"], "song_sha256": s["sha256"],
            "song_file_size_bytes": s["file_size_bytes"],
            "speech_relative_path": p["relative_path"], "speech_sha256": p["sha256"],
            "speech_file_size_bytes": p["file_size_bytes"],
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    ok = len(out_rows) == 1012
    print("wrote %d paired rows to %s" % (len(out_rows), OUT_PATH))
    print("GATE_3_PAIR_KEY:", "PASS" if ok else "FAIL (row count != 1012)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
