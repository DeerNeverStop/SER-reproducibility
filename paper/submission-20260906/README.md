# ICASSP 2027 manuscript review packet / 两版论文审阅包

Prepared for Tian Xie's requested Claude review on 2026-09-06.
This is a manuscript-only review snapshot, not a conference submission, a public
release, or authorization to change the experimental record.

## Start here

- [English manuscript PDF](english/xie.pdf): 4 technical pages plus a fifth page
  containing references only, using the unmodified ICASSP 2027 `spconf.sty`.
- [English LaTeX source](english/main.tex).
- [Chinese explanatory PDF](chinese/explainer.pdf): 8 pages explaining the same
  experiments, statistics, findings, and limitations in accessible Chinese.
- [Editable Chinese text](chinese/中文解读.md).
- [Claude review request](CLAUDE_REVIEW_REQUEST.md): scope and expected feedback.
- [Source bundle](source_bundle.zip): the original local source/checklist package.
- [Submission checklist and build notes](README_交付与投稿检查.md).
- [PDF structure QA](pdf_qa.json) and [upload manifest](UPLOAD_MANIFEST.json).

**These are the current review manuscripts.** Older files at `paper/main.tex`,
`paper/v2/main.tex`, and `paper/v2/main.pdf` remain untouched for provenance; do not
mistake them for this revised submission draft.

Author details supplied by the user: Tian Xie, University of Toronto,
tianjack.xie@mail.utoronto.ca. Author approval, ORCID, and any actual funding or
ethics disclosures still need to be settled before conference submission.

## Where the experimental results live

No frozen results are replaced by this packet. The branch starts from
`4fe0bb52f15b56a6626363be31a378b0f9661293`, the N14R2 execution specification commit.

- [Completed N14R2 result release](https://github.com/DeerNeverStop/SER/releases/tag/SER26-N14R2-complete-20260906):
  17 uploaded assets, including the closed-run archive, original and corrected
  scores, original failed and corrected verification reports, locks, hashes,
  metadata erratum, report, and replay instructions.
- [Original v2 results in this tree](../../v2/evidence/main/results/results.json)
  and [verification](../../v2/evidence/main/results/verification.json).
- [v2 experimental discussion](https://github.com/DeerNeverStop/SER/pull/6).

The repository is PRIVATE at upload time. Being uploaded to this repository does
not make the manuscripts or experimental material publicly accessible. It also
does not establish that the pre-execution receipt was public before execution.
The manuscripts disclose that provenance limitation and use the conservative
description "pre-execution frozen" / "prespecified," not a positive claim of
public preregistration. Historical tags containing `prereg` are retained as
historical names only.

## Scope and integrity

Only this new manuscript directory is added. Experimental source, plans, locks,
scores, existing releases, tags, and old manuscripts are not edited. No raw audio,
feature cache, checkpoint archive, credentials, font files, compiler binary, or
compiler cache is included here. Existing experimental archives remain linked
from the release instead of being duplicated in Git history.

All 11 uploaded original deliverables are byte-identical to the locally reviewed
files; `UPLOAD_MANIFEST.json` records their SHA-256 values. The English and Chinese
PDFs were rendered and visually checked page by page, and both passed the local
structural checks. These checks do not certify scientific correctness or author
approval; Claude is being asked for an independent substantive review.

The original QA JSON retains its local generation paths. In this repository,
`output/pdf/xie.pdf` maps to `english/xie.pdf`, and
`output/pdf/SER_中文解读.pdf` maps to `chinese/explainer.pdf`; the bytes are unchanged.
The source bundle intentionally preserves the local generation-stage notes,
including their statement that no upload had been performed in that earlier
stage. This README records the later private GitHub upload for review.

See the source checklist for build dependencies. The standalone build tools are
included as text; neither rebuilding the PDFs nor running experiments is required
to review the supplied PDFs and sources.
