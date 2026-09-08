# P4-D1 license & source verification (RAVDESS Speech–Song matched contrast)

Written: 2026-08-16, 00:00 hourly band (Claude). Independent CPU pre-check,
gates 2–4 of `P4_D1_NEXT_EXPERIMENT_ASSIGNMENT_FOR_CLAUDE.md` §5.

## Source

- **Corpus**: RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech
  and Song), Livingstone & Russo.
- **Official archive**: Zenodo record DOI `10.5281/zenodo.1188976`.
- **License**: CC BY-NC-SA 4.0 (as documented for the Speech half in this
  project's `P1_PREREGISTRATION.md`; the Song half is part of the same
  Zenodo record and ships under the same license — see provenance
  cross-check below, which confirms both zips are literally packaged
  together in the same record bundle).
- **Research use**: permitted (non-commercial); this project is
  non-commercial academic research. Attribution is required — the paper
  must cite the original RAVDESS release.
- **No new network download was performed this session.** Both audio
  archives (`Audio_Song_Actors_01-24.zip`, and the RAVDESS speech tree
  already extracted at `E:\科研\SER\data`) were already present locally
  from prior work; this session only re-verified and re-hashed them.

## Archive identity

| File | SHA-256 |
|---|---|
| `E:\科研\SER\Audio_Song_Actors_01-24.zip` (standalone, used for extraction) | `89f3df66cc67ebb70c8af6f72f3f359bee497f6f763a9d863b6477619716d428` |
| `Audio_Song_Actors_01-24.zip` entry inside `E:\科研\SER\1188976.zip` (the full Zenodo record 1188976 bundle, also present locally) | `89f3df66cc67ebb70c8af6f72f3f359bee497f6f763a9d863b6477619716d428` — **byte-identical**, confirming the standalone file used for D1 is the genuine record 1188976 artifact, not a stray/edited copy |

The full-record bundle `1188976.zip` also independently corroborates the
"Actor 18 has no Song" correction already recorded in the D1 assignment
(§3): its `Video_Song_Actor_*.zip` entries run 01–17, 19–24 — Actor 18 is
absent from Song in both the audio and video halves of the same Zenodo
record, cross-validating the extracted-audio finding below via a second,
independent artifact.

## Extraction

- Destination (E drive, per data-placement rule): `E:\claudework_data\ICASSP2027-corpora\ravdess-song\`
- Extracted WAV count: **1,012** (matches the zip's own WAV entry count exactly; 24 non-WAV entries were the 24 top-level `Actor_NN/` directory markers, not data).
- Extracted actor directories: **23** — `Actor_01`..`Actor_17`, `Actor_19`..`Actor_24`. `Actor_18` is absent from the archive itself, not merely empty.
- RAVDESS speech source (unchanged, read-only, not re-downloaded): `E:\科研\SER\data\Actor_01`..`Actor_24\` (24 actors, 1,440 files) — this is the same tree `P1_PREREGISTRATION.md` already documented and hashed for P1; D1 does **not** modify or re-extract it.

## Independent recount (gates 2–3)

Produced by `07-ser-repro-protocol-audit/tools/build_d1_file_manifest.py`
(re-walks both raw sources from scratch; does not trust the pre-existing P1
`ravdess_manifest.csv` numbers) and
`07-ser-repro-protocol-audit/tools/build_d1_pair_manifest.py`:

- Song: 1,012 files parsed, 0 unparsed, 23 distinct actors, 0 read errors.
- Speech (raw, all 24 actors × 8 emotions): 1,440 files parsed, 0 unparsed, 0 read errors.
- Speech restricted to the common six emotions (neutral/calm/happy/sad/angry/fearful) and Actor 18 dropped: **1,012 files**, 23 distinct actors, 0 read errors.
  - Dropped for Actor 18: 60 files.
  - Dropped for disgust/surprised (not in Song's emotion set): 368 files.
- Pair key `(actor, emotion, intensity, statement, repetition)`: 0 duplicate keys on either side; **0 keys only-in-song; 0 keys only-in-speech**; joined `pair_manifest.csv` has exactly **1,012** rows.

Full output: `file_manifest.csv` (2,024 rows, both channels), `pair_manifest.csv` (1,012 rows).

## Anomaly registry (gate 4)

Produced by `build_d1_anomaly_registry.py`, full library SHA-256 + amplitude
scan over all 2,024 files (song + restricted speech):

- **1 duplicate-SHA256 group, 2 files, both on the Speech side**:
  `Actor_07/03-01-03-01-02-01-07.wav` and `Actor_07/03-01-03-01-02-02-07.wav`
  (actor 07, happy, intensity normal, statement 02, repetitions 01 vs 02 —
  byte-identical content under two different repetition labels). This is
  **not new**: it independently reproduces the exact same anomaly already
  on record for P1/P5 (`results/mechanism_exposure/RECON_NOTES.md:34`),
  which is a strong cross-check that this session's independent re-hash is
  correct. Not cross-actor, not cross-label, not cross-channel.
- All-zero files: 0. Extremely-short (<0.5s) files: 0. Unreadable files: 0.

**Frozen handling rule (written before any D1 result is ever computed,
matching P1 precedent of not deleting real corpus duplicates):** the one
Actor-07 duplicate pair is **kept** in the main analysis — it is a real,
official-archive attribute (same as the RAVDESS/CREMA-D duplicates P1 kept),
it does not collapse two different pair keys (repetition 01 and 02 remain
distinct keys), and it cannot bias the Song–Speech pairing since it is a
within-Speech content duplicate, not a cross-channel or cross-actor
collision. No sensitivity re-run is triggered by this anomaly alone. If a
future scan (post-loader) surfaces an anomaly with cross-actor, cross-label,
or cross-channel structure, that would require re-opening this freeze
before any GPU run — none does at this time.

## Status against the D1 ten-gate checklist (assignment §5)

| Gate | Status |
|---|---|
| 1 (task book on disk) | already PASS (prior band) |
| 2 (license + archive hash + 1,012/1,012 + 23×44) | **PASS this band** |
| 3 (pair key one-to-one, bidirectional diff = 0) | **PASS this band** |
| 4 (label mapping + anomaly registry frozen) | **PASS this band** |
| 5 (two-layer split hash definition) | already PASS (prior band, text only) |
| 6 (speaker exposure / train-test intersection = 0) | **not started** — needs outer/inner split assignment, which needs the reused 02/10 canonical-key + actor-fold assets (assignment §3.1) located and re-verified first |
| 7 (OOF exactly-once, zero index overlap) | **not started**, same dependency as gate 6 |
| 8 (normalization/augmentation train-fold-only policy) | **not started** |
| 9 (loader unit tests + mutation test, fail-closed) | **partially done this band**: `test_d1_precheck.py` covers actor-ID uniqueness, label-mapping completeness, pair-key uniqueness, and two fail-closed mutation tests (corrupted pair key; fabricated actor-split collision) — all 12 assertions PASS. Still missing: a mutation test against the actual split manifest (gate 5/6), since that manifest doesn't exist yet. |
| 10 (`nvidia-smi` + GPU headroom check) | last step before training, not yet due |

**GPU GO is still NOT granted.** Gates 6–8 and the split-manifest half of
gate 9 remain for a subsequent band.
