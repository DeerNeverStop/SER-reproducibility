# ICASSP 2027 paper artifact

This folder contains the manuscript build provenance assembled from the frozen P0/P1 audit outputs and the D1 matched-panel diagnostic. It does **not** rerun training or modify any result file.

## Status and template caveat

- Target: ICASSP 2027 regular paper.
- The official 2027 author kit was not published on the conference site as of 2026-08-15.
- The generated PDF therefore follows the current official constraints (US Letter, two columns, 9 pt minimum body text, no page numbers, four technical pages plus a fifth references-only page) and the public ICASSP 2026 `spconf` conventions as a provisional layout.
- Before submission, replace the provisional source wrapper with the official 2027 kit and rerun the official PDF compliance check.
- ICASSP 2027 uses single-anonymous review. Replace `[AUTHOR NAME]`, `[AFFILIATION]`, and `[EMAIL]` in the generated source/PDF, and make the author list exactly match the submission form.

Official links:

- [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/)
- [ICASSP 2027 publishing options](https://2027.ieeeicassp.org/authors/publishing-and-paper-presentation-options/)
- [Provisional ICASSP 2026 paper kit](https://cmsworkshops.com/ICASSP2026/papers/paper_kit.php)

## Files

- `../PRIOR_ART_COLLISION_MATRIX.md`: risk-ranked collision search and the resulting claim boundary.
- `build_demo.py`: reads frozen CSV/JSON evidence, asserts the approved counts, generates the figure, fact catalog, Markdown, provisional LaTeX, and final PDF.
- `paper_demo.md`: generated readable manuscript.
- `main_provisional.tex`: generated provisional `spconf` source.
- `main_numbercheck.tex`: mechanically derived technical-body view used by the semantic number checker. Only bibliography metadata is removed; every scientific claim remains.
- `paper_facts_07.json`: generated fact catalog for number auditing.
- `number_audit.md` / `number_audit.json`: generated report from the existing project-04 number checker.
- `icassp2027_protocol_audit_m2.pdf`: the frozen five-page artifact PDF.

The frozen deliverable is `icassp2027_protocol_audit_m2.pdf`. The older demo PDF and rendered page images are intentionally omitted from this paper-focused Git tree.

## Frozen evidence inputs

- `results/execution_validation/tier_summary.csv`
- `results/protocol_premium/run_status_summary.json`
- `results/protocol_premium/summary/verification.json`
- `results/protocol_premium/summary/protocol_metric_summary.csv`
- `results/protocol_premium/summary/protocol_premium.csv`
- `protocols/p1/P1_PREREGISTRATION.md`
- `results/ravdess-mode-matched/scoring/d1_scoring_report.json`
- `results/ravdess-mode-matched/training/completion_manifest_v6.json`

The build fails closed if the audit counts, fit count, OOF completeness, verification status, or eight Random-GroupKFold UAR rows differ from the frozen contract.

## Rebuild

Use the bundled primary-runtime Python because it contains ReportLab and pypdf:

```powershell
& 'C:\Users\jock8\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\build_demo.py
```

Then run the semantic number checker on the scientific body. Bibliographic years, volume numbers, page ranges, and DOI digits are outside this audit and were instead verified against the linked primary records in `PRIOR_ART_COLLISION_MATRIX.md`.

```powershell
& 'E:\科研\SER\ser_gpu\Scripts\python.exe' `
  'E:\科研\SER\codex\tools\check_paper_numbers.py' `
  --facts .\paper_facts_07.json `
  --json-out .\number_audit.json `
  --markdown-out .\number_audit.md `
  .\main_numbercheck.tex
```

## Scientific boundary

This artifact intentionally excludes P2, P5, and the later Stage-B multi-corpus campaign. It reports the P0 final 20/5/5 adjudication, P1 controlled protocol effects, and the compact D1 matched-panel diagnostic only. `yes` means speaker exclusivity was not enforced by the observed split mechanism (or was violated on the frozen synthetic manifest); it is not proof of the exact speaker overlap in a paper's reported run.
