# Original supplementary audit script

`duration_quality_audit.py` is preserved byte-for-byte from the execution-time
audit, matching the source SHA in its JSON result. It reads only the plan,
manifest and existing audio-quality receipts; it does not open E2 results.

Its constants reflect the original local directory layout. Configure those
input/output paths before rerunning it elsewhere; do not point its output at
immutable input evidence. The archived CSV/receipt and result are under
`../data/audio_budget/`. The independently checked report additionally clarifies
that relative differences are averaged within matched pairs, and that this audit
reuses the extractor's verified waveform identities rather than rereading audio.

The main result verification/replay tools with explicit CLI inputs are in
`../../audit_tools/`.
