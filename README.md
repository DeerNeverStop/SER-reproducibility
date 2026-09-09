# Speech emotion recognition: validation, data design and reproducible results

Research artifacts accompanying **Validation Speaker Exposure and Checkpoint Selection in Speech Emotion Recognition**, by Tian Xie. This is a research manuscript and public reproducibility snapshot, not a claim of ICASSP acceptance.

**中文入口：** [全部实验及成果，用人话解释](docs/public-release-20260908/EXPERIMENTS_ZH.md) · [公开材料与复现说明](docs/public-release-20260908/REPRODUCING.md)

**Latest revision:** [v2026.09.09](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.09) updates manuscript presentation and adds an explicitly post hoc window figure. No training or primary test results change. See the [revision notes and scope](docs/public-release-20260909/README.md). The [v2026.09.08 release](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.08) remains the source for its original downloadable bundles.

## Start here

| What you want | Entry point |
|---|---|
| Latest manuscript revision | [English PDF](paper/review-revision-20260909/english/xie.pdf), [editable LaTeX](paper/review-revision-20260909/english/main.tex), [complete absolute-results supplement](paper/review-revision-20260909/ABSOLUTE_RESULTS.md) |
| Previous manuscript and results explanation | [2026-09-08 English PDF](paper/supplement-results-20260908/english/xie.pdf), [its LaTeX source](paper/supplement-results-20260908/english/main.tex), [Chinese results report](docs/autodl-supplement-20260908/RESULTS_REPORT_ZH.md) |
| Complete research history | [Experiment inventory](docs/public-release-20260908/EXPERIMENTS_ZH.md): completed, exploratory, superseded, interrupted and unexecuted work are distinguished |
| Original 384 formal fits | [Results and interpretation](paper/final-20260907/RESEARCH_REPORT_中文.md), [all score files](v3/final_program_20260907/reports/scores/), [frozen scientific design](v3/final_program_20260907/SCIENCE_DESIGN.md) |
| Separate 720-fit supplement | [Protocol](docs/autodl-supplement-20260908/README.md), [all results](docs/autodl-supplement-20260908/results/), [code and portable plans](v3/autodl_supplement_20260908/README.md) |
| CPU-only verification | [Reproduction levels and commands](docs/public-release-20260908/REPRODUCING.md), including all 6 original and all 10 supplemental primary tests |
| Downloadable artifacts | [2026-09-09 revision notes](docs/public-release-20260909/README.md), [existing v2026.09.08 bundles](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.08), [their file identities and availability](docs/public-release-20260908/AVAILABILITY.md) |
| Data and reuse terms | [Rights and original dataset sources](RIGHTS_AND_DATA.md) |

## What the current paper establishes

The original **384 formal fits** comprise 360 main trajectories and 24 paired group-role controls. A separate **720-fit** supplement adds HuBERT on three corpora and longer WavLM candidate windows on two. Technical pilots are excluded. The current paper therefore uses **1,104 formal fits**; this is neither the whole project's training total nor 1,104 independent people.

Each main trajectory is shared by four checkpoint-selection rules: seen/unseen validation speakers crossed with cross-entropy/UAR. All four selected models face the same unfamiliar-speaker outer test. The results support criterion- and window-dependent selection effects within these programs. They do not establish that one validation rule is universally best, that voice identity is the sole mechanism, or that speaker-related differences have been eliminated.

All original six and supplemental ten primary tests are retained in their **separate** multiplicity families, including imprecise and unfavorable results. Pointwise intervals are conditional on the fixed corpora and programs. The underlying corpora and panels have prior research exposure. Private pre-execution freezes are not relabeled as public preregistrations by this release.

## Quick numerical check — no GPU or audio

```text
git clone https://github.com/DeerNeverStop/SER-reproducibility.git
cd SER-reproducibility
python -m pip install -r docs/public-release-20260908/requirements-replay.txt
python docs/public-release-20260908/verify_public_manifest.py --repo .
python docs/public-release-20260908/verify_public_results.py --repo .
```

The small verifier independently recomputes the 16 primary tests from published draw tables. It does not authenticate unpublished logits or rerun training. The release also provides the original 384-fit prediction bundle for a deeper CPU replay; [instructions](docs/public-release-20260908/REPRODUCING.md) specify the extra dependencies and scope. The 720-fit raw prediction and retained-weight archive remains controlled; its complete reported draw, unit, checkpoint-selection and curve tables are public.

## Historical programs

The inventory also covers the public-code audit, P1 split protocols, speech/song diagnostics, v2 speaker/text controls, N14R2, deployment/calibration studies, the 1,440-fit recording-allocation study, the separate 720-fit speaker-selection study, 480-fit dual validation, and zero-new-fit reanalyses. Synthetic speech stopped at a technical probe; a validated synthetic-augmentation experiment was not completed. Superseded N14 inference and the interrupted N14R run remain identified as such.

Historical reports are preserved for provenance and may describe their original dates, paths or private release references. Use this page and the experiment inventory for current status. The original `artifact_manifest.json` describes a historical 607-file tree, not this public snapshot; use `PUBLICATION_MANIFEST.json` for the present release.

## Publication boundaries

This repository starts from a curated snapshot of the private research workspace. It does not expose the private repository's history, PR conversations, cloud accounts or operational records. Retained Python training/scoring code and scientific result tables preserve their bytes. Selected narrative documents are explicitly marked public projections, with distinct hashes where private operational details are omitted. The original private repository and full local archives remain separate.

The 2026-09-09 manuscript revision distinguishes the public numerical artifact from the controlled full archive. Earlier manuscripts retain their historical artifact references; use the current navigation and versioned release notes for public availability. The earlier Chinese PDF is a historical manuscript and is not represented as a full translation of the latest English revision.

No raw audio, model weights, credentials, third-party research-paper PDFs or bundled conference style are included. See [rights](RIGHTS_AND_DATA.md), [availability](docs/public-release-20260908/AVAILABILITY.md), and [CITATION.cff](CITATION.cff).
