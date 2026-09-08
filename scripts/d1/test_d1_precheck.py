"""P4-D1 CPU pre-check loader tests (assignment doc section on "loader tests":
must cover speaker-ID uniqueness, label-mapping completeness, and at least
one pair-key/actor-split mutation test that fails closed).

T1  speaker-ID uniqueness: exactly 23 distinct actors per channel, and the
    two channels' actor sets are identical (both exclude Actor 18).
T2  label-mapping completeness: only the six common emotion codes (1-6)
    appear in either channel; no disgust(7)/surprised(8) leaked through the
    restricted-speech filter; every code has a non-empty, correct name.
T3  pair-key one-to-one: pair_manifest.csv has exactly 1,012 rows and no
    (actor, emotion, intensity, statement, repetition) key repeats.
T4  bidirectional match, clean data: `verify_bidirectional_match` returns
    (True, "") on the real, unmutated key sets loaded from file_manifest.csv.
T5  mutation test (fail-closed): corrupting a single song-side pair key
    (bumping its repetition field so it no longer matches its speech
    counterpart) must make `verify_bidirectional_match` return False with a
    non-empty, actor-identifying diff -- not silently pass.
T6  mutation test (fail-closed), actor-split flavor: duplicating one actor's
    song key set under a fabricated actor id (simulating an actor-split
    leak / mislabel) must also be caught as an asymmetry, not silently
    absorbed by a partial/any-match check.

This module reads only file_manifest.csv / pair_manifest.csv already
produced by build_d1_file_manifest.py / build_d1_pair_manifest.py in this
same directory; it does not touch raw audio, run any GPU code, or open any
test-label file (none exists for D1 -- D1's target is a paired premium
computed after real training, not a sealed key).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# D1's canonical result root is under ICASSP2027/, not this topic directory's
# own results/ (see P4_D1_NEXT_EXPERIMENT_ASSIGNMENT_FOR_CLAUDE.md section 7).
D1_RESULT_ROOT = Path(r"E:\科研\ICASSP2027\results\corpora\ravdess-mode-matched")
MANIFEST_PATH = D1_RESULT_ROOT / "file_manifest.csv"
PAIR_MANIFEST_PATH = D1_RESULT_ROOT / "pair_manifest.csv"

COMMON_SIX = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful"}

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load_manifest_rows():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pair_rows():
    with open(PAIR_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_of(row):
    return (
        int(row["actor"]), int(row["emotion_code"]), int(row["intensity"]),
        int(row["statement"]), int(row["repetition"]),
    )


def verify_bidirectional_match(song_keys, speech_keys):
    """Returns (ok: bool, detail: str). Fail-closed: any duplicate on either
    side, or any asymmetry in the two key sets, is reported as not ok."""
    song_set, speech_set = set(song_keys), set(speech_keys)
    if len(song_set) != len(song_keys):
        return False, "song has duplicate pair keys"
    if len(speech_set) != len(speech_keys):
        return False, "speech has duplicate pair keys"
    only_song = song_set - speech_set
    only_speech = speech_set - song_set
    if only_song or only_speech:
        return False, (
            f"asymmetric: {len(only_song)} keys only in song "
            f"(e.g. {sorted(only_song)[:3]}), {len(only_speech)} only in speech "
            f"(e.g. {sorted(only_speech)[:3]})"
        )
    return True, ""


def main() -> int:
    rows = load_manifest_rows()
    song = [r for r in rows if r["channel"] == "song"]
    speech = [r for r in rows if r["channel"] == "speech"]

    # T1: speaker-ID uniqueness / cross-channel identity
    song_actors = sorted(set(int(r["actor"]) for r in song))
    speech_actors = sorted(set(int(r["actor"]) for r in speech))
    check("T1a song has exactly 23 distinct actors", len(song_actors) == 23, str(len(song_actors)))
    check("T1b speech(restricted) has exactly 23 distinct actors", len(speech_actors) == 23, str(len(speech_actors)))
    check("T1c actor 18 absent from both channels", 18 not in song_actors and 18 not in speech_actors)
    check("T1d song and speech actor sets are identical", song_actors == speech_actors,
          f"song={song_actors[:3]}..{song_actors[-3:]} speech={speech_actors[:3]}..{speech_actors[-3:]}")

    # T2: label-mapping completeness, no leakage of disgust/surprised
    song_emotions = sorted(set(int(r["emotion_code"]) for r in song))
    speech_emotions = sorted(set(int(r["emotion_code"]) for r in speech))
    check("T2a song emotion codes == common six {1..6}", song_emotions == sorted(COMMON_SIX), str(song_emotions))
    check("T2b speech(restricted) emotion codes == common six {1..6}", speech_emotions == sorted(COMMON_SIX),
          str(speech_emotions))
    names_ok = all(r["emotion_name"] == COMMON_SIX[int(r["emotion_code"])] for r in song + speech)
    check("T2c every row's emotion_name matches its code's canonical name", names_ok)

    # T3: pair-key one-to-one in the already-built pair manifest
    pair_rows = load_pair_rows()
    pair_keys = [
        (int(r["actor"]), int(r["emotion_code"]), int(r["intensity"]), int(r["statement"]), int(r["repetition"]))
        for r in pair_rows
    ]
    check("T3a pair_manifest.csv has exactly 1012 rows", len(pair_rows) == 1012, str(len(pair_rows)))
    check("T3b pair_manifest.csv keys are unique", len(set(pair_keys)) == len(pair_keys),
          f"{len(pair_keys)} rows, {len(set(pair_keys))} unique keys")

    # T4: bidirectional match on real (clean) data
    song_keys = [key_of(r) for r in song]
    speech_keys = [key_of(r) for r in speech]
    ok, detail = verify_bidirectional_match(song_keys, speech_keys)
    check("T4 clean data: bidirectional match holds", ok, detail)

    # T5: mutation test -- corrupt one song key's repetition field
    mutated_song_keys = list(song_keys)
    victim = mutated_song_keys[0]
    corrupted = (victim[0], victim[1], victim[2], victim[3], 3 if victim[4] != 3 else 4)  # invalid repetition value
    mutated_song_keys[0] = corrupted
    ok, detail = verify_bidirectional_match(mutated_song_keys, speech_keys)
    check("T5 mutation (corrupted pair key) is caught, fail-closed", ok is False and bool(detail), detail)

    # T6: mutation test -- fabricate a duplicate actor by relabeling one
    # song actor's block under an actor id that collides with another
    # actor's existing key set (simulates an actor-split identity leak).
    mutated_song_keys_2 = list(song_keys)
    donor_actor = song_actors[0]
    target_actor = song_actors[1]
    mutated_song_keys_2 = [
        (target_actor, k[1], k[2], k[3], k[4]) if k[0] == donor_actor else k
        for k in mutated_song_keys_2
    ]
    ok, detail = verify_bidirectional_match(mutated_song_keys_2, speech_keys)
    check("T6 mutation (actor-split collision) is caught, fail-closed", ok is False and bool(detail), detail)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
