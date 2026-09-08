"""P4-D1: anomaly_registry.csv — full-library SHA-256 duplicate scan plus
all-zero / extremely-short amplitude checks, over the D1 file set (song +
restricted speech, 2,024 files from file_manifest.csv).

Per the D1 assignment section 3: "Actor 07 等重复 hash、不可解码、全零、
极短、跨标签/跨演员同 hash 都须在训练前进入 anomaly registry，并冻结主
分析与敏感性处理" — this script only detects and records; it does not
delete or resample anything, and freezing the handling rule is done in the
accompanying markdown note, not by silently dropping rows here.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

MANIFEST_PATH = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched\file_manifest.csv"
OUT_PATH = r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched\anomaly_registry.csv"

SPEECH_ROOT = r"E:\科研\SER\data"
SONG_ROOT = r"E:\claudework_data\ICASSP2027-corpora\ravdess-song"

SHORT_DURATION_SECONDS = 0.5  # flag anything shorter than this for review


def abs_path(row):
    root = SONG_ROOT if row["channel"] == "song" else SPEECH_ROOT
    return os.path.join(root, row["relative_path"].replace("/", os.sep))


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    anomalies = []

    # 1) duplicate SHA-256 groups (within-channel or cross-channel)
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r)
    dup_groups = 0
    for h, members in by_hash.items():
        if len(members) < 2:
            continue
        dup_groups += 1
        actors = sorted(set(m["actor"] for m in members))
        emotions = sorted(set(m["emotion_name"] for m in members))
        channels = sorted(set(m["channel"] for m in members))
        cross_actor = len(actors) > 1
        cross_label = len(emotions) > 1
        cross_channel = len(channels) > 1
        for m in members:
            anomalies.append({
                "anomaly_type": "duplicate_sha256",
                "channel": m["channel"],
                "relative_path": m["relative_path"],
                "actor": m["actor"],
                "emotion_name": m["emotion_name"],
                "sha256": m["sha256"],
                "group_size": len(members),
                "cross_actor": cross_actor,
                "cross_label": cross_label,
                "cross_channel": cross_channel,
                "detail": "group members: " + "; ".join(
                    "%s/%s" % (x["channel"], x["relative_path"]) for x in members
                ),
            })

    # 2) all-zero / extremely-short amplitude scan (reads every file once)
    zero_count = 0
    short_count = 0
    unreadable_count = 0
    for r in rows:
        p = abs_path(r)
        try:
            data, samplerate = sf.read(p, dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            unreadable_count += 1
            anomalies.append({
                "anomaly_type": "unreadable",
                "channel": r["channel"], "relative_path": r["relative_path"],
                "actor": r["actor"], "emotion_name": r["emotion_name"],
                "sha256": r["sha256"], "group_size": "", "cross_actor": "",
                "cross_label": "", "cross_channel": "", "detail": str(exc),
            })
            continue

        max_abs = float(np.max(np.abs(data))) if data.size else 0.0
        duration_s = (data.shape[0] / samplerate) if samplerate else 0.0

        if max_abs == 0.0:
            zero_count += 1
            anomalies.append({
                "anomaly_type": "all_zero",
                "channel": r["channel"], "relative_path": r["relative_path"],
                "actor": r["actor"], "emotion_name": r["emotion_name"],
                "sha256": r["sha256"], "group_size": "", "cross_actor": "",
                "cross_label": "", "cross_channel": "",
                "detail": "max_abs_sample=0.0, duration_s=%.3f" % duration_s,
            })
        if duration_s < SHORT_DURATION_SECONDS:
            short_count += 1
            anomalies.append({
                "anomaly_type": "extremely_short",
                "channel": r["channel"], "relative_path": r["relative_path"],
                "actor": r["actor"], "emotion_name": r["emotion_name"],
                "sha256": r["sha256"], "group_size": "", "cross_actor": "",
                "cross_label": "", "cross_channel": "",
                "detail": "duration_s=%.3f (< %.2fs threshold)" % (duration_s, SHORT_DURATION_SECONDS),
            })

    fieldnames = [
        "anomaly_type", "channel", "relative_path", "actor", "emotion_name",
        "sha256", "group_size", "cross_actor", "cross_label", "cross_channel", "detail",
    ]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anomalies)

    print("scanned %d files" % len(rows))
    print("duplicate sha256 groups: %d (rows flagged: %d)" % (
        dup_groups, sum(1 for a in anomalies if a["anomaly_type"] == "duplicate_sha256")))
    print("all-zero files: %d" % zero_count)
    print("extremely-short files (<%.2fs): %d" % (SHORT_DURATION_SECONDS, short_count))
    print("unreadable files: %d" % unreadable_count)
    print("wrote %d anomaly rows to %s" % (len(anomalies), OUT_PATH))
    print("GATE_4_ANOMALY_REGISTRY: DONE (registry records anomalies; it does not gate pass/fail by itself)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
