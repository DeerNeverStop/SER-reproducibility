# VoiceDesign: fixed 100-attempt technical probe

This checks whether four repeated fictional adult English voice descriptions retain a recognizable identity across text and emotion changes. It does **not** establish a 24-speaker E3 dataset, implement voice cloning, train SER, or establish that the generated voices are consistent identities. A repeated VoiceDesign description does not guarantee the same perceived person. The four generated neutral anchors are comparison audio only; no anchor is supplied to the generator.

The probe has four fixed descriptions (two adult female, two adult male), six intended emotions, and four text prompts: `4 × 6 × 4 = 96` probe calls, plus one extra neutral anchor call per description, totaling **100 calls at most**. The anchor repeats the first of the four training sentences with its own fixed seed. It is an independent rendition rather than a duplicate file. All calls use VoiceDesign, including anchors; there is no model selection or best-of-N search.

The four texts are the lexicographically first four prompt codes in the allowed eight training prompts of the frozen coverage plan's U/CNN/fold 0/rotation 0/draw 0 unit. The code verifies the source plan and unit hashes, eight training/six stop/two query prompt roles, and exclusion of that context's query prompts. The immutable manifest includes the chosen codes, literal English sentences, voice descriptions, instructions, seeds, runtime configuration, input plan hashes, model identity, and probe source SHA. These are local-context training texts: a sentence can have a query role in a different rotation. This probe provides no SER score and must not tune a later global experiment using held-out SER results.

Sentence wording comes from the [official CREMA-D README](https://github.com/CheyneyComputerScience/CREMA-D#sentences). The literal mapping and probe source are hashed; the mutable README URL itself is not claimed to be a frozen Git revision. The [official Qwen API](https://github.com/QwenLM/Qwen3-TTS#voice-design) supports `generate_voice_design(text=..., language='English', instruct=..., ...)`. The Base model uses a different voice-cloning API: its lack of independently documented instruction control must not be treated as permission to pass VoiceDesign emotion instructions to Base. A later VoiceDesign → Base cloning workflow would be a separate, prereviewed design.

## Commands

Run from the repository root using the isolated environment already provisioned by the operator. These commands never install packages or download model files.

```bash
PY=/workspace/ser-coverage-tts-env/bin/python
$PY -m v3.speaker_coverage.tts.probe freeze \
  --plan /path/to/coverage/plan.json \
  --model-dir /workspace/coverage-models/qwen-voice-design \
  --model-pins /workspace/ops-coverage/tts_model_pins.json \
  --out /workspace/tts-probe-100

$PY -m v3.speaker_coverage.tts.probe generate \
  --manifest /workspace/tts-probe-100/manifest.json \
  --model-dir /workspace/coverage-models/qwen-voice-design \
  --model-pins /workspace/ops-coverage/tts_model_pins.json

$PY -m v3.speaker_coverage.tts.probe blind \
  --manifest /workspace/tts-probe-100/manifest.json
```

`freeze` performs CPU file hashing and writes the manifest and all 100 per-attempt configs. `generate` is the only GPU command. The source must remain byte-identical to the frozen source; editing it requires a separate probe directory and disclosure of the change. Do not silently replace a manifest after observing outputs. `blind` requires all 100 attempts to have terminal receipts, writes empty rating forms, and refuses to overwrite a previous rating export.

The model pin file must contain `model_id`, an immutable 40-character `revision`, `package: "qwen-tts==0.1.1"`, and `files: {relative_path: {bytes, sha256}}`. Every model file outside `.cache` is checked against that inventory before freeze and before generation, including the tokenizer/configuration files included in the snapshot. The manifest separately binds the pin-file bytes and the canonical snapshot inventory. There is no external reference audio to hash; generated anchor WAV hashes are recorded in their receipts and the private blind key. A missing file, a changed source, or an altered completed WAV is an error, never a reason to download or regenerate silently.

## Fixed runtime and receipts

Generation uses one local model and one GPU (`cuda:0`), `torch.bfloat16`, SDPA, batch size one, and no more than 512 new tokens. The other fixed sampling arguments are in each config. Each attempt sets Python, NumPy, Torch, and CUDA seeds. Fixed seeds do not promise bitwise reproducibility across library, hardware, or kernel versions; runtime versions and GPU identity are logged.

Each `attempts/aNNN/` contains a frozen `config.json`, a `started.json` written before the API call, and a terminal `receipt.json`. Successful numeric mono output is saved as `audio.wav` with floating-point samples, without amplitude normalization, clipping repair, quality filtering, or quality-driven replacement. Duration, wall time, peak allocated/reserved VRAM, finiteness, RMS, peak amplitude, and fraction of samples with absolute amplitude at least 0.999 are logged. Silence, clipping, bad emotion, or a short utterance do not trigger resampling. Non-finite/empty/non-mono output or an exception is a failed attempt with a missing WAV explicitly recorded. These engineering checks are not human quality judgments.

Both successful and failed attempts are terminal and skipped on resume. An attempt with a start record but no terminal receipt is marked interrupted and is not called again; an orphan WAV, if present, is hashed but not treated as a successful item. This may yield fewer than 100 successful files, which is reported. An OS process lock prevents concurrent generation into the same directory and releases on process exit. If model loading fails before an attempt starts, no generation attempt has been consumed; the operator can resolve the infrastructure issue and resume with the same frozen inputs. Do not delete terminal receipts to obtain a preferred sample.

## Pending independent human evaluation

Give each of three independent human raters only their own `blind/rater_N/` folder. Do not share the manifest, attempt configs, `blind_private_key.json`, or other raters' answers. Each folder contains randomized anonymous audio filenames, independently ordered quality and identity forms, and instructions. Quality fields ask perceived emotion (including OTHER/UNCLEAR), naturalness, and intelligibility. Identity items compare each non-anchor clip with the successfully generated neutral anchors, allowing NONE/UNCLEAR. Correct matches and intended labels exist only in the private key. Missing anchors limit the identity task and are explicitly reported.

All answer fields are empty. The summary says `awaiting_three_independent_human_raters`, with zero completed raters; creating these forms is not completion of human evaluation. This is a small engineering diagnostic with four described voices and one anchor per voice; ratings do not by themselves establish 24 independent speakers or general emotion/identity controllability. A single atypical anchor can distort matching; report that limitation rather than replacing it after listening. The identity task also cannot separate all effects of timbre, emotion, and prosody.

The probe script itself does not run ASV; its original blind-export summary records that boundary. The separate `tts/asv.py` diagnostic pins the existing ECAPA snapshot and waveform preprocessing, compares each successful non-anchor clip against all four neutral anchors, and reports agreement with the intended voice description. This is a proxy affected by emotion, prosody and the choice of one anchor; it is not verified human identity or emotion quality. Its separate `asv_diagnostic/summary.json` records the later execution status. No ASV or human outcome from this probe alters the frozen E2 U/R/C panels or their inference rules.

Tests use synthetic metadata, fake model bytes, and an explicitly fake CPU backend in temporary directories. They are software checks, not real generation or evidence of model/corpus feasibility.
