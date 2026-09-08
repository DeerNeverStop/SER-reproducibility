# Artifact availability and exact identities

Public snapshot version: `v2026.09.08`. This is a publication of existing research artifacts, not a new training run, new preregistration or claim of conference acceptance.

## Included in the repository

The source snapshot contains the current English manuscript and editable source, Chinese results explanations, historical experiment navigation, scientific training/scoring implementations, fixed plans/split metadata, complete compact numerical results, figures and CPU verification scripts. `PUBLICATION_MANIFEST.json` records every published file's size and SHA-256, the private source-workspace commit used for preparation, copied-source identities, publication adaptations and exclusions. It does not inherit the source repository's old branches or PR history.

The latest supplement retains all ten primary tests and all eight model/corpus/window reporting groups. The original six-test family remains separate. New 720-fit outputs contain 960 unit-window rows, 312 draw rows, 104 descriptive rows, 4,800 selected-epoch rows and all 18,000 epoch-curve rows. These reporting rows must not be counted as new fits or independent speakers.

## Versioned download

[Release v2026.09.08](https://github.com/DeerNeverStop/SER-reproducibility/releases/tag/v2026.09.08)

| Attachment | Contents | Bytes | SHA-256 |
|---|---|---:|---|
| `formal_numeric_bundle.zip` | Original 384-fit prediction replay bundle, without raw audio or model weights | 59,897,568 | `c3f14905050f13401788a926d21dbb754cf28031499bfa77abef913315b14e6f` |

Additional small attachments, including the manuscript and recipe package, are listed in the Release's `SHA256SUMS.txt`. The original 384 ZIP was checked for members and sensitive-material candidates, then actually extracted and replayed using this public source directory: 384 formal units, 63,901 comparisons, maximum absolute error `4.440892098500626e-16`, no checkpoint inference or training. Its original ledger and failure records are preserved; a historical gate is not a fresh audit of omitted weight bytes. Minor author-computer paths remain as provenance, so the bundle is not anonymous.

The [Study II recipe ZIP](../../v3/data_recipes_20260907/releases/study2_recording_recipes_v1.zip) is also versioned in the repository. It reconstructs the original recording lists and 1,440 configurations. It is not a new audio dataset or additional training evidence.

## Controlled or omitted material

- Raw audio, feature caches, voice embeddings, pretrained bases and fine-tuned weights must be obtained under the original providers' terms or through an appropriately authorized archive arrangement. They are not included here.
- The 720-fit per-epoch logits and retained-checkpoint archive is not a Release attachment. Full published curves and draw/selection tables support result inspection, while the lightweight replay only verifies the published table layer.
- Older studies have different public payload coverage. The experiment inventory links to their source and compact evidence; a historical local path or private GitHub Release reference does not promise public download of the corresponding large archive.
- Provider account balances, live infrastructure identities, shell controllers, private assessment/conversation branches and third-party research-paper PDFs are not published. Selected narrative documents have explicitly labeled public projections; frozen scientific result tables and code are not silently edited to sanitize them.
- The bundled third-party conference style and historical source ZIPs containing it are omitted. Obtain the style from the official kit, as described in [rights](../../RIGHTS_AND_DATA.md).

Original private source-commit identifiers and original artifact hashes describe provenance. Public access begins with this new release; those identifiers do not establish that the material was publicly visible before execution. Hash agreement detects byte changes relative to an inventory, not authenticity, independent retraining or statistical validity.

See [REPRODUCING.md](REPRODUCING.md) for concrete commands and their limits.
